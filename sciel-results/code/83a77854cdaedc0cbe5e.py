"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition with deadline guard."""
    # Deadline check: if we're close to the limit, return a safe fallback
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using numerically stable computation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    cost = np.maximum(cost, 1e-9)
    score = ei / cost
    
    # Add exploration bonus based on uncertainty
    # Use a weighted combination to balance exploitation and exploration
    exploration_weight = 0.3
    exploration_bonus = exploration_weight * sigma / cost
    
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
