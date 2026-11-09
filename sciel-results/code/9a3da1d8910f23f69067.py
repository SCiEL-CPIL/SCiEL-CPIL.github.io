import numpy as np
import time

# Module-level state to track evaluation count for adaptive beta
_acq_call_count = 0
_acq_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _acq_call_count
    _acq_call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _acq_start_time
    if elapsed > 300.0:
        # Return UCB with moderate exploration as fallback
        sigma_safe = np.where(sigma > 0, sigma, 1e-6)
        scores = -mu + 1.0 * sigma_safe
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # GP-LCB variant with adaptive exploration coefficient
    # beta increases as t (number of evaluations) grows to expand search radius
    # t ranges from ~6 to ~30 in this problem
    t = _acq_call_count
    beta = 1.0 + 0.1 * t
    
    # Ensure sigma is finite and positive
    sigma_safe = np.where(sigma > 0, sigma, 1e-6)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # But acquisition function returns scores where HIGHER is better
    # So we negate: score = -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Handle NaNs and Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
