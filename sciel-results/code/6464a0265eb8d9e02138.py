import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Context-free EMA state for each arm
        # High initial values to encourage exploration of unvisited arms
        # This acts as a strong optimistic bias
        self.emas = np.full(n_arms, 10.0) 
        
        # Aggressive forgetting factor to track non-stationary changes quickly
        # 0.90 means the effective window is ~10 samples
        self.gamma = 0.90
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # 5s safety margin for 120s limit
        
        # Random seed for tie-breaking or potential future stochasticity, 
        # though argmax with unique floats is deterministic.
        # We use deterministic argmax for speed.
        
    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # If we are past deadline, just pick the first arm to be safe/fast
            return 0
            
        # Select arm with highest estimated reward (EMA)
        # np.argmax returns the first index in case of ties
        return int(np.argmax(self.emas))

    def update(self, context, arm, reward):
        # Update the EMA for the selected arm
        # EMA_new = gamma * EMA_old + (1 - gamma) * reward
        self.emas[arm] = self.gamma * self.emas[arm] + (1.0 - self.gamma) * reward
