"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()
    deadline = 330.0 - 5.0  # safety margin before 330s kill

    # Cost-aware acquisition: balance improvement, uncertainty, and cost
    # Use a gamma parameter to control exploration vs exploitation
    gamma = 0.5  # exploration weight
    
    # Compute z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Uncertainty bonus (UCB-like)
    ucb = gamma * sigma
    
    # Combined acquisition: (EI + UCB) / cost
    # This balances expected improvement and exploration, normalized by cost
    combined = ei + ucb
    
    # Normalize by cost (cost is 1 + mean(x), so in [1, 2])
    scores = combined / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're getting close to the limit, return immediately
    if time.time() - t_start > deadline * 0.8:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
