"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_t0 = time.time()
_DEADLINE = 295.0  # safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition with adaptive exploration.
    
    Uses a hybrid: UCB for exploration/exploitation balance,
    scaled by a cost-aware factor that's less aggressive than 1/cost.
    The cost factor uses sqrt(1/cost) to moderately penalize expensive
    points without completely suppressing cheap exploration.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return mu-based score (exploit best predictions)
        return -mu / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost-aware scaling: use power law for smoother cost penalty
    # sqrt(1/cost) is less aggressive than 1/cost, allowing cheap points
    # to still be competitive while preferring them
    cost_factor = 1.0 / np.sqrt(cost + 1e-9)
    
    # Adaptive UCB beta: start with moderate exploration, 
    # the GP uncertainty naturally adapts
    beta = 2.0
    
    # UCB: mu + beta * sigma (we want to minimize, so use -mu + beta*sigma)
    # Higher score = more promising
    ucb = -mu + beta * sigma
    
    # Cost-aware UCB score
    score = ucb * cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
# EVOLVE-BLOCK-END
