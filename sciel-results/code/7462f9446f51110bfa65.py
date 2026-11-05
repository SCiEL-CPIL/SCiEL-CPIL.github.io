import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.dim = dim
        self.gamma = 0.90
        self.alpha = 1.0
        # Per-arm Exponential Moving Averages of rewards
        self.ema_reward = np.zeros(self.n)
        # Per-arm effective visit counts (decayed)
        self.effective_visits = np.ones(self.n)
        # Global effective time step
        self.effective_time = 1.0

    def select(self, context):
        # Ignore context
        log_t = np.log(self.effective_time + 1e-8)
        # Use UCB-style exploration bonus
        # Bonus is higher for arms with fewer effective visits
        exploration = self.alpha * np.sqrt(log_t / (self.effective_visits + 1e-8))
        scores = self.ema_reward + exploration
        return int(np.argmax(scores))
    
    def update(self, context, arm, reward):
        # Ignore context
        # Update EMA of reward
        self.ema_reward[arm] = self.gamma * self.ema_reward[arm] + (1 - self.gamma) * reward
        # Update effective visits
        self.effective_visits[arm] = self.gamma * self.effective_visits[arm] + 1
        # Update global effective time
        self.effective_time = self.gamma * self.effective_time + 1
