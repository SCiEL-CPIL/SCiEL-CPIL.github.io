"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # to ensure we don't timeout. The evaluator allows 300s per eval, 
    # but the total budget is 330s. 
    # 25 steps * 300s = 7500s potential, but we only have 330s total.
    # So on average we have ~13s per step.
    # If elapsed > 320s, we are in danger.
    if time.time() - _start_time > 320:
        # Fallback to a simple greedy score
        return -mu 

    # Adaptive UCB
    # t is the number of optimization steps taken (excluding initial 5)
    # _call_count is 1 for the first optimization step
    t = _call_count
    
    # Decay beta
    # Start with 1.5, decay with power 0.5
    # beta = 1.5 / np.sqrt(1 + t)
    # Let's try a slower decay or different start
    # Constant UCB 2.0 failed. EI (score 1.0) is baseline.
    # Let's try UCB with beta starting at 1.0 and decaying.
    beta = 1.0 / (1.0 + 0.1 * t)
    
    # UCB for minimization: mu - beta * sigma
    # Higher score is better.
    # We want to maximize (mu - beta * sigma) ? No, we want to minimize f.
    # So we want to pick x that minimizes f.
    # UCB score for minimization: S(x) = mu(x) - beta * sigma(x).
    # We return S(x). The evaluator picks the x with the highest S(x).
    # So we want S(x) to be high when mu is low (good) and sigma is high (uncertain).
    
    scores = mu - beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
