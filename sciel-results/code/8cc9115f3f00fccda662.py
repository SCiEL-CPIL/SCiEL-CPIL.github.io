import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Context-free Multi-Armed Bandit with Discounted UCB
        # We ignore the context vector entirely.
        # This is robust against non-stationarity and computationally cheap.
        
        self.gamma = 0.90  # Forgetting factor. 0.90 gives a window of ~10 steps.
        
        # mu[a]: Discounted mean reward for arm a
        # N[a]: Discounted count of pulls for arm a
        # Initialize with neutral values to avoid division by zero and allow quick start
        self.mu = np.zeros(n_arms)
        self.N = np.ones(n_arms)  # Start with 1 to avoid div by zero, acts as a prior
        
        # Exploration parameter
        # Standard UCB uses sqrt(2 * log(t) / n).
        # We will use a simplified version: alpha * sqrt(log(t_total) / N[a])
        # alpha controls the exploration-exploitation trade-off.
        # A larger alpha means more exploration.
        self.alpha = 1.0
        
        self.t_total = 0.0  # Total effective time

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback to greedy
            return int(np.argmax(self.mu))

        # Calculate UCB scores
        # Ensure N[a] is not zero (initialized to 1, so it won't be)
        # log(t_total) needs to be positive. Start t_total at 1.
        if self.t_total < 1:
            self.t_total = 1.0
            
        log_t = np.log(self.t_total + 1)  # +1 to avoid log(0) if t=0, though we start at 1
        
        # UCB = mu + alpha * sqrt(2 * log_t / N)
        # The factor 2 is standard in UCB1 theory.
        scores = self.mu + self.alpha * np.sqrt(2.0 * log_t / self.N)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t_total += 1.0
        
        # Update discounted mean
        # mu_new = gamma * mu_old + (1 - gamma) * reward
        self.mu[arm] = self.gamma * self.mu[arm] + (1.0 - self.gamma) * reward
        
        # Update discounted count
        # N_new = gamma * N_old + 1
        # This effectively weights the current sample as 1, and previous samples decay by gamma.
        self.N[arm] = self.gamma * self.N[arm] + 1.0

