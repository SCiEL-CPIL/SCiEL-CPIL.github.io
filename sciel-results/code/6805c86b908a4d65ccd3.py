import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize weights for each arm: E[reward * context]
        # We maintain an EMA of x * r
        self.weights = [np.zeros(dim) for _ in range(n_arms)]
        
        # Decay factor for the EMA. 
        # A lower alpha means a shorter effective window, faster adaptation to changes.
        # 0.95 corresponds to a window of roughly 20 steps.
        self.alpha = 0.95
        
        # Epsilon-greedy parameters
        # Start with high exploration, decay to low exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback: Greedy
            context_vec = np.array(context)
            scores = [np.dot(self.weights[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return int(np.random.randint(self.n))
        else:
            # Exploit: arm with highest predicted reward
            scores = [np.dot(self.weights[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context_vec = np.array(context)
        
        # Update the EMA for the selected arm
        # weights[arm] = alpha * weights[arm] + (1 - alpha) * reward * context_vec
        # This approximates the conditional expectation E[r | x, arm] scaled by feature density,
        # which is sufficient for ranking arms in a linear setting if features are consistent.
        self.weights[arm] = self.alpha * self.weights[arm] + (1 - self.alpha) * reward * context_vec
