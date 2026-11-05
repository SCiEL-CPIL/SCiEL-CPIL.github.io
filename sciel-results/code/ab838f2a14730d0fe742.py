"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Policy:
    """Discounted Linear Bandit with UCB."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.gamma = 0.95  # Discount factor for exponential forgetting
        self.beta = 2.0    # Exploration parameter
        self.lambda_reg = 1.0 # Initial regularization
        
        # Initialize A and b for each arm
        # A is d x d, b is d x 1
        self.A = [self.lambda_reg * np.eye(self.d) for _ in range(self.n)]
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        
    def select(self, context):
        context = np.asarray(context).flatten()
        scores = []
        for a in range(self.n):
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                A_inv = np.eye(self.d) / self.lambda_reg
                
            theta = A_inv @ self.b[a]
            pred = theta @ context
            var = context @ A_inv @ context
            score = pred + self.beta * np.sqrt(max(var, 0))
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context).flatten()
        reward = float(reward)
        
        # Update with discounting
        # A_a <- gamma * A_a + x x^T
        # b_a <- gamma * b_a + r x
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
# EVOLVE-BLOCK-END
