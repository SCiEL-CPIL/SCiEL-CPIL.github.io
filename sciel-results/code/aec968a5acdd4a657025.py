import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # UCB for minimization: we want low mu and high sigma.
        # Score = gamma * sigma - mu
        # Cost aware: divide by cost
        gamma = 2.5
        ucb_scores = gamma * sigma - mu
        # Avoid division by zero or negative costs (cost is always >= 1)
        scores = ucb_scores / (cost + 1e-9)
        # Ensure finite
        return np.where(np.isfinite(scores), scores, 0.0)
