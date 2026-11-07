import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 320:  # 10s safety margin
        # Return scores that favor the lowest-cost points (safe fallback)
        return -cost  # Higher score for lower cost
    
    # Thompson Sampling: draw one sample from the posterior
    # sample = mu + sigma * z, where z ~ N(0,1)
    z = np.random.standard_normal(len(mu))
    sample = mu + sigma * z
    
    # Score: higher is better. We want low objective per unit cost.
    # score = -sample / cost
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-9)
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
