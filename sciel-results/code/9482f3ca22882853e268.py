"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 290.0 # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to simple EI
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI: divide by cost
    # Cost is 1 + mean(x scaled to [0, 1]), so range is [1, 2]
    # We want to maximize EI / cost
    score = ei / cost
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps when EI is zero (mu >= f_best) but sigma is high
    # Use a smooth, bounded term: sigma / (sigma + 0.1)
    # This is 0 when sigma=0 and approaches 1 as sigma->inf
    # Scale it by a small factor to not dominate EI
    exploration_bonus = 0.1 * sigma / (sigma + 0.1)
    
    # Combine: EI/cost + small exploration bonus
    # The exploration bonus is not divided by cost to ensure exploration
    # even in high-cost regions, but it's small enough not to dominate
    score = score + exploration_bonus
    
    return score
# EVOLVE-BLOCK-END
