"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # Safety margin: if we've used 295s, switch to pure exploitation
    # to ensure we finish the current evaluation and potentially the next
    # without timing out, prioritizing the best known point.
    if elapsed > 295:
        # Exploit: maximize -mu / cost
        # cost is in [1, 2]
        scores = -mu / cost
        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Standard Cost-Aware Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    # Clip z to avoid numerical issues in cdf/pdf for very large values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring
    scores = ei / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
