import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Context-free Multi-Armed Bandit with Exponential Moving Average (EMA)
        # Structural change: Ignore the linear context 'x' entirely.
        # Model: reward ~ N(mu_a, sigma^2) where mu_a changes over time.
        # Estimator: EMA of observed rewards for each arm.
        # This is robust to non-stationarity if the forgetting factor is tuned correctly.
        # It is computationally trivial (O(n) per step).
        
        # Aggressive forgetting factor to track drift quickly.
        # 0.9 means ~10% weight on new data, 90% on past. Effective window ~10 steps.
        self.gamma = 0.90
        
        # Store EMA of rewards for each arm.
        # Initialized to 0.0.
        self.reward_ema = np.zeros(n_arms)
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        self.t = 0

    def select(self, context):
        # Check deadline to avoid timeout
        if time.time() > self.deadline:
            # Fallback: Greedy selection
            return int(np.argmax(self.reward_ema))
            
        self.t += 1
        
        # Exploration strategy: Add Gaussian noise to the scores.
        # This is a form of "noisy greedy" or "linear TS with scalar posterior".
        # The noise magnitude is fixed to ensure continuous exploration,
        # which is crucial in non-stationary settings to detect changes.
        # A smaller sigma reduces exploration, larger increases it.
        # 0.1 is a reasonable starting point for unit-scale rewards.
        noise = np.random.normal(0, 0.1, self.n)
        
        scores = self.reward_ema + noise
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update the EMA for the selected arm.
        # S_t = gamma * S_{t-1} + (1 - gamma) * r_t
        
        # This update is O(1).
        self.reward_ema[arm] = self.gamma * self.reward_ema[arm] + (1 - self.gamma) * reward
