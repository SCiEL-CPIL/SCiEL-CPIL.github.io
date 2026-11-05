"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Context-Free Non-Stationary Multi-Armed Bandit Policy.
    
    Strategy:
    - Ignores context vector entirely.
    - Maintains per-arm Exponential Moving Average (EMA) of rewards.
    - Fixed aggressive forgetting factor (gamma=0.90) to track non-stationarity.
    - Epsilon-Greedy exploration with fixed epsilon.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize reward estimates for each arm to 0.
        # In a Gaussian bandit, rewards are often centered around 0 or small values.
        # Starting at 0 is a neutral prior.
        self.arm_values = np.zeros(n_arms)
        
        # Hyperparameters
        # Forgetting factor for EMA. 
        # Lower gamma means faster forgetting of old data, better for non-stationarity.
        # 0.90 is aggressive, allowing quick adaptation to weight changes.
        self.gamma = 0.90
        
        # Epsilon for exploration.
        # Fixed epsilon to ensure continuous exploration of all arms,
        # which is critical when the best arm can change due to non-stationarity.
        self.epsilon = 0.2
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        # Deadline check: if we are running out of time, just return a random arm 
        # to avoid hanging in complex logic.
        if self._check_deadline():
            return int(np.random.randint(self.n))

        # Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n))
        
        # Greedy: select arm with highest estimated value
        # arm_values is a vector of shape (n,)
        return int(np.argmax(self.arm_values))

    def update(self, context, arm, reward):
        # Deadline check
        if self._check_deadline():
            return

        # Update the exponential moving average for the pulled arm
        # EMA_t = gamma * EMA_{t-1} + (1 - gamma) * reward_t
        self.arm_values[arm] = self.gamma * self.arm_values[arm] + (1 - self.gamma) * reward
# EVOLVE-BLOCK-END
