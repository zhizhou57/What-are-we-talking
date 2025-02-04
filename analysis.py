import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from collections import defaultdict
import openai
import json
import os
from openai import OpenAI
from tqdm import tqdm
import multiprocessing
import logging
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class DataSource(Enum):
    QQ = "qq"
    WECHAT = "wechat"

# Constants and Configuration
@dataclass
class Config:
    IDLE_HOURS: int = 6
    MIN_MESSAGES_FOR_ANALYSIS: int = 3
    API_BASE_URL: str = "<openai_api_base_url>"
    MODEL_NAME: str = "gpt-4o"
    OUTPUT_DIR: str = "output"
    DATA_SOURCE: DataSource = DataSource.QQ

class OpenAIClient:
    """Handles all OpenAI API interactions"""
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_summary(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的对话分析师，善于提取对话的核心内容和情感。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=16384
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise

class ConversationManager:
    """Handles conversation splitting and management"""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def split_conversations(self, idle_hours: int) -> List[Dict]:
        conversations = []
        current_conversation = []
        prev_time = None
        
        for _, row in self.df.iterrows():
            current_time = row['DateTime']
            msg_content = row['MsgContent']
            
            # 解码消息内容
            if 'DecodedMsg' in row and pd.notna(row['DecodedMsg']):
                msg_content = row['DecodedMsg']
            
            if prev_time is not None:
                time_diff = (current_time - prev_time).total_seconds() / 3600  # 转换为小时
                
                if time_diff > idle_hours:
                    if current_conversation:
                        conversations.append({
                            'messages': current_conversation,
                            'start_time': current_conversation[0]['time'],
                            'end_time': current_conversation[-1]['time']
                        })
                    current_conversation = []
            
            current_conversation.append({
                'time': current_time,
                'sender': row['SenderUin'],
                'content': msg_content
            })
            prev_time = current_time
        
        # 添加最后一个对话
        if current_conversation:
            conversations.append({
                'messages': current_conversation,
                'start_time': current_conversation[0]['time'],
                'end_time': current_conversation[-1]['time']
            })
        
        return conversations

class ChatAnalyzer:
    def __init__(self, data_file: str, data_source: DataSource = DataSource.QQ, output_dir: str = Config.OUTPUT_DIR):
        """Initialize chat analyzer with better error handling and logging"""
        self.output_dir = Path(output_dir)
        self.data_source = data_source
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.output_dir / 'analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        try:
            self.df = self._load_data(data_file)
            self.conversation_manager = ConversationManager(self.df)
            self.openai_client = None  # Will be initialized when needed
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise

    def _load_data(self, data_file: str) -> pd.DataFrame:
        """Load data based on source type"""
        if self.data_source == DataSource.QQ:
            return self._load_qq_data(data_file)
        elif self.data_source == DataSource.WECHAT:
            return self._load_wechat_data(data_file)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def _load_qq_data(self, csv_file: str) -> pd.DataFrame:
        """Load and prepare QQ CSV data"""
        df = pd.read_csv(csv_file)
        df['DateTime'] = pd.to_datetime(df['Time'], unit='s')
        df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        return df

    def _load_wechat_data(self, data_file: str) -> pd.DataFrame:
        """Load and prepare WeChat data"""
        df = pd.read_csv(data_file)
        
        # Convert CreateTime to datetime and localize to Shanghai timezone
        df['DateTime'] = pd.to_datetime(df['CreateTime'], unit='s')
        df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        
        # Map message types to content
        def get_message_content(row):
            if row['Type'] == 1:  # Text message
                return row['StrContent']
            elif row['Type'] == 3:  # Image
                return '[图片]'
            elif row['Type'] == 47:  # Emoji
                return '[表情]'
            elif row['Type'] == 49:  # System message
                return '[系统消息]'
            else:
                return '[其他类型消息]'
        
        df['MsgContent'] = df.apply(get_message_content, axis=1)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Sender': 'SenderUin',
            'StrTime': 'Time'
        })
        
        # Filter relevant columns
        df = df[['DateTime', 'SenderUin', 'MsgContent']]
        
        return df

    def initialize_openai(self, api_key: str) -> None:
        """Initialize OpenAI client"""
        self.openai_client = OpenAIClient(api_key, Config.API_BASE_URL)

    def analyze_conversations(self, idle_hours: int = Config.IDLE_HOURS) -> List[Dict]:
        """Analyze conversations with proper error handling"""
        try:
            conversations = self.conversation_manager.split_conversations(idle_hours)
            self._save_conversations(conversations)
            return conversations
        except Exception as e:
            logging.error(f"Error analyzing conversations: {e}")
            raise

    def _save_conversations(self, conversations: List[Dict]) -> None:
        """Save conversation data to file"""
        output_path = self.output_dir / 'conversations.json'
        try:
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(self._format_conversations_for_save(conversations), f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving conversations: {e}")
            raise

    def _format_conversations_for_save(self, conversations: List[Dict]) -> List[Dict]:
        """Format conversation data for JSON serialization"""
        formatted_conversations = []
        for conv in conversations:
            formatted_messages = []
            for msg in conv['messages']:
                formatted_msg = {
                    'time': msg['time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'sender': msg['sender'],
                    'content': msg['content']
                }
                formatted_messages.append(formatted_msg)
            
            formatted_conv = {
                'messages': formatted_messages,
                'start_time': conv['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': conv['end_time'].strftime('%Y-%m-%d %H:%M:%S')
            }
            formatted_conversations.append(formatted_conv)
        return formatted_conversations

    def summarize_conversation(self, conversation, openai_api_key):
        """使用ChatGPT总结对话内容"""
        if not self.openai_client:
            self.initialize_openai(openai_api_key)
        
        # 格式化对话内容
        formatted_messages = []
        for msg in conversation['messages']:
            sender = "用户A" if msg['sender'] == self.df['SenderUin'].iloc[0] else "用户B"
            formatted_messages.append(f"{sender}: {msg['content']}")
        
        conversation_text = "\n".join(formatted_messages)
        
        # 构建提示
        prompt = f"""请简要总结以下对话的主题（限制在30字以内）：

对话时间：{conversation['start_time']} 到 {conversation['end_time']}

{conversation_text}

总结："""
        
        try:
            return self.openai_client.generate_summary(prompt)
        except Exception as e:
            return f"总结失败: {str(e)}"

    def analyze_chat_topics(self, conversations, openai_api_key):
        if os.path.exists(os.path.join(self.output_dir, 'chat_topics.json')):
            with open(os.path.join(self.output_dir, 'chat_topics.json'), 'r', encoding='utf-8') as f:
                return json.load(f)
        """分析聊天话题"""
        summaries = []
        for conv in tqdm(conversations):
            # 过滤掉消息数量太少的对话
            if len(conv['messages']) < Config.MIN_MESSAGES_FOR_ANALYSIS:
                continue
            
            summary = self.summarize_conversation(conv, openai_api_key)
            summaries.append({
                'start_time': conv['start_time'],
                'end_time': conv['end_time'],
                'message_count': len(conv['messages']),
                'summary': summary
            })

            print(f"从{conv['start_time']}到{conv['end_time']},总结成功: {summary}")
        
        # 保存话题分析结果
        with open(os.path.join(self.output_dir, 'chat_topics.json'), 'w', encoding='utf-8') as f:
            json.dump([{
                'start_time': summary['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': summary['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'message_count': summary['message_count'],
                'summary': summary['summary']
            } for summary in summaries], f, ensure_ascii=False, indent=2)
        
        return summaries

    def generate_activity_chart(self, idle_hours: int = Config.IDLE_HOURS) -> None:
        """Generate activity chart showing monthly message counts from raw data"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 直接从原始DataFrame按月统计消息数
        monthly_counts = defaultdict(int)
        for _, row in self.df.iterrows():
            date = row['DateTime']
            month_key = date.strftime('%Y-%m')  # 格式如 "2024-01"
            monthly_counts[month_key] += 1
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        months = list(monthly_counts.keys())
        counts = list(monthly_counts.values())
        
        plt.bar(range(len(months)), counts, width=0.8)
        plt.title('每月消息数量统计')
        plt.xlabel('月份')
        plt.ylabel('消息数量')
        
        # 设置x轴刻度和标签
        plt.xticks(range(len(months)), months, rotation=45, ha='right')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'chat_activity.png'))
        plt.close()

    def analyze_chat_patterns(self, summaries: List[Dict]) -> str:
        """Analyze chat patterns and generate comprehensive insights"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        # Format summaries for analysis
        formatted_data = []
        for summary in summaries:
            formatted_data.append(
                f"时间：{summary['start_time']} - {summary['end_time']}\n"
                f"消息数：{summary['message_count']}\n"
                f"主题：{summary['summary']}\n"
            )
        
        analysis_prompt = (
            "请分析以下聊天记录数据，总结出：\n"
            "1. 两人讨论的主要话题（按照重要度排序）\n"
            "2. 两人的性格特点和互动模式\n"
            "3. 聊天的时间规律（如果有）\n\n"
            f"聊天记录数据：\n{chr(10).join(formatted_data)}\n\n"
            "请用中文回答，控制在400字以内"
        )

        try:
            return self.openai_client.generate_summary(analysis_prompt)
        except Exception as e:
            logging.error(f"Chat pattern analysis failed: {e}")
            return "分析失败：无法生成聊天模式分析"

    def generate_html_report(self, summaries: List[Dict]) -> None:
        """Generate an HTML report with chat analysis results"""
        # Get chat pattern analysis
        chat_analysis = self.analyze_chat_patterns(summaries)
        
        # Convert markdown to HTML (basic conversion)
        chat_analysis_html = chat_analysis.replace('\n\n', '</p><p>')
        chat_analysis_html = f'<p>{chat_analysis_html}</p>'
        
        template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>我们的聊天故事</title>
                <style>
                    body {{ 
                        font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif; 
                        max-width: 1200px; 
                        margin: 0 auto; 
                        padding: 20px; 
                    }}
                    .conversation-card {{
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 10px 0;
                        background: white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .analysis-section {{
                        background: #f9f9f9;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 20px 0;
                    }}
                    .timeline {{ margin: 20px 0; }}
                    .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .chart {{ 
                        margin: 20px auto;
                        width: 50%;
                        text-align: center;
                    }}
                    .chart img {{
                        max-width: 100%;
                        height: auto;
                        display: block;
                        margin: 0 auto;
                    }}
                </style>
            </head>
            <body>
                <h1>✨ 我们的聊天故事 ✨</h1>
                <div class="stats">
                    <div>💬 总对话数: {total_conversations}</div>
                    <div>📅 时间跨度: {date_range}</div>
                </div>
                <div class="analysis-section">
                    <h2>🔍 聊天大揭秘</h2>
                    {chat_analysis}
                </div>
                <div class="chart">
                    <h2>📈 聊天活跃度</h2>
                    <img src="{img_path}" alt="聊天活跃度" />
                    <p>看看我们什么时候聊得最嗨！</p>
                </div>
                <div class="timeline">
                    <h2>💭 那些年我们聊过的天</h2>
                    {conversation_cards}
                </div>
                <div style="text-align: center; margin-top: 30px;">
                    <p>💖 每一句对话都是我们故事的碎片 💖</p>
                </div>
            </body>
            </html>
        """
        
        conversation_cards = ""
        for summary in summaries:
            card = f"""
                <div class="conversation-card">
                    <h3>⏰ {summary['start_time']} - {summary['end_time']}</h3>
                    <p>📨 消息数量: {summary['message_count']}</p>
                    <p>💡 对话主题: {summary['summary']}</p>
                </div>
            """
            conversation_cards += card
        
        img_path = os.path.join('.', 'chat_activity.png')
        
        html_content = template.format(
            total_conversations=len(summaries),
            date_range=f"{summaries[0]['start_time']} 至 {summaries[-1]['end_time']}",
            conversation_cards=conversation_cards,
            img_path=img_path,
            chat_analysis=chat_analysis_html
        )
        
        with open(os.path.join(self.output_dir, 'report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)

    def export_to_pdf(self) -> None:
        """Convert HTML report to PDF"""
        try:
            import pdfkit
            pdfkit.from_file(
                os.path.join(self.output_dir, 'report.html'),
                os.path.join(self.output_dir, 'report.pdf')
            )
        except Exception as e:
            logging.error(f"PDF generation failed: {e}")

def main():
    try:
        # Load configuration from environment or config file
        config = {
            'csv_file': 'wechat_messages.csv',
            'openai_api_key': '<你的openai_api_key>',
            'output_dir': 'output_wx',
            'data_source': DataSource.WECHAT
        }

        # Initialize analyzer
        analyzer = ChatAnalyzer(config['csv_file'], config['data_source'], config['output_dir'])
        analyzer.initialize_openai(config['openai_api_key'])

        # Run analysis pipeline
        conversations = analyzer.analyze_conversations()
        summaries = analyzer.analyze_chat_topics(conversations, config['openai_api_key'])

        # Generate reports
        analyzer.generate_activity_chart()
        analyzer.generate_html_report(summaries)
        analyzer.export_to_pdf()
        
        logging.info("Analysis and reports generated successfully")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()