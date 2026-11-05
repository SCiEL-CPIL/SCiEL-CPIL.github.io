"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

class Policy:
    """Sliding Window Linear Bandit."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window_size = 50
        self.lam = 1.0  # Regularization parameter, larger for stability with small windows
        self.hists = [deque(maxlen=self.window_size) for _ in range(n_arms)]

    def select(self, context):
        scores = []
        for a in range(self.n):
            if len(self.hists[a]) == 0:
                # No data for this arm, high uncertainty
                # Return a high value to encourage exploration initially?
                # Or 0? Standard LinUCB with empty A is undefined.
                # Let's return a large constant to force exploration of unseen arms.
                scores.append(10.0)
                continue
            
            X = np.array([x for x, _ in self.hists[a]])
            Y = np.array([y for _, y in self.hists[a]])
            
            # Compute A and b
            A = X.T @ X + self.lam * np.eye(self.d)
            b = X.T @ Y
            
            try:
                inv_A = np.linalg.inv(A)
                theta = inv_A @ b
                pred = theta @ context
                var = context @ inv_A @ context
                # Exploration parameter
                # As window fills, uncertainty decreases. 
                # We might want a constant beta or one that scales.
                beta = 1.5
                scores.append(pred + beta * np.sqrt(var))
            except np.linalg.LinAlgError:
                scores.append(0.0)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.hists[arm].append((context, reward))
# EVOLVE-BLOCK-END
