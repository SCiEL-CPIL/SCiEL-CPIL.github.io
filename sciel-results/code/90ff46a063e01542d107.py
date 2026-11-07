"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_ACQ_START_TIME = time.time()
_ACQ_TIME_LIMIT = 330.0
_ACQ_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Linear UCB approach.
    Maximizes (f_best - mu + kappa * sigma) / cost.
    """
    # Deadline guard
    if time.time() - _ACQ_START_TIME > _ACQ_TIME_LIMIT - _ACQ_SAFETY_MARGIN:
        # Return a neutral score to allow the loop to finish safely
        return np.zeros_like(mu)

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Cost-aware UCB score
    # We want to maximize this value.
    # f_best - mu: Exploitation (distance to current best)
    # kappa * sigma: Exploration (uncertainty)
    # Dividing by cost: Cost efficiency
    kappa = 1.0
    numerator = f_best - mu + kappa * sigma
    scores = numerator / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
