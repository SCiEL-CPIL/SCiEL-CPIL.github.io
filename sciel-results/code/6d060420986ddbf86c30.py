"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290.0  # 10s safety margin

def _check_deadline():
    return time.time() - _start_time > _TIMEOUT

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        return np.ones_like(mu)
    
    # Ensure sigma is positive for calculations
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # Avoid numerical issues with extreme z
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize -mu + beta * sigma
    beta = 2.0
    ucb_score = -mu + beta * sigma
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Normalize scores to [0, 1] to make them comparable
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
        
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb_score)
    
    # 4. Hybrid Score
    # Blend EI and UCB. EI is strong for exploitation, UCB for exploration.
    # Weight 0.5 for each.
    hybrid = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # 5. Cost Awareness
    # Divide by cost to favor cheaper evaluations
    # Cost is in [1, 2]
    scores = hybrid / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
