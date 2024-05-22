from langchain_groq import ChatGroq
from dotenv import load_dotenv
from crewai import Agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.get_thread import GmailGetThread
from langchain_community.tools import DuckDuckGoSearchRun
from textwrap import dedent
from tools import CreateDraftTool

load_dotenv()

class EmailPersonalizationAgents():
    def __init__(self):
        self.gmail = GmailToolkit()
        # self.llm = ChatGroq(model="mixtral-8x7b-32768")
        self.llm = ChatOpenAI(model = "gpt-3.5-turbo")

    def personalize_email_agent(self):
                return Agent(
            role="Email Personalizer",
            goal="""
                Personalize template emails for recipients using their information.
                Given a template email and recipient information (name, email, bio, last conversation), 
                personalize the email by incorporating the recipient's details 
                into the email while maintaining the core message and structure of the original email. 
                This involves updating the introduction, body, and closing of the email to make 
                it more personal and engaging for each recipient.
                """,
            backstory="""
                As an Email Personalizer, you are responsible for customizing template emails for individual recipients based on their information and previous interactions.
                """,
            verbose=True,
            llm=self.llm,
            max_iter=5,
            cache = False   
        )

    def ghostwriter_agent(self):
                return Agent(
            role="Ghostwriter",
            goal="""
                Revise draft emails to adopt the Ghostwriter's writing style.
                Use an informal, engaging, and slightly sales-oriented tone, mirroring the Ghostwriter's final 
                email communication style.
                """,
            backstory="""
                As a Ghostwriter, you are responsible for revising draft emails to match the Ghostwriter's 
                writing style, focusing on clear, direct communication with a friendly and approachable tone.
                """,
            verbose=True,
            llm=self.llm,
            max_iter=5,
            tools = [
                    DuckDuckGoSearchRun(),
                    GmailGetThread(api_resource=self.gmail.api_resource)
                    ],
            allow_delegation = False
        )
    
    def email_response_writer(self):
            return Agent(
                    role = "Email Response Writer",
                    goal = "Draft responses to action-required emails",
                    backstory = dedent(""" You are a skilled writer, adept at crafting clear, concise, and effective
                                       email. Your strength lies in your ability to communicate effectively,
                                       ensuring that every message tailored to address the specific
                                        needs and context of the email."""),
                    tools = [
                            DuckDuckGoSearchRun(),
                            GmailGetThread(api_resource=self.gmail.api_resource),
                            CreateDraftTool.create_draft
                    ],
                    verbose = True,
                    llm = self.llm,
                    allow_delegation = False
                    )
    
    