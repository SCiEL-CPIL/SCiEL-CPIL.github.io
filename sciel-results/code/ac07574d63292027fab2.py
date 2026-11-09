import numpy as np
from scipy.stats import norm
import time

# Track call index to adapt strategy over the optimization loop
_call_count = [0]
_start_time = [None]

def acquisition(mu, sigma, f_best):
    _call_count[0] += 1
    step = _call_count[0]
    
    # Time guard: if we're getting close to the 330s limit, return a safe result
    if _start_time[0] is None:
        _start_time[0] = time.time()
    elapsed = time.time() - _start_time[0]
    if elapsed > 320:  # Safety margin
        # Return a neutral score to avoid timeout
        return np.zeros_like(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Adaptive UCB strategy:
    # The key insight is that we need to balance exploration and exploitation.
    # Early on (steps 1-5), we should explore more to find promising regions.
    # Later (steps 6-25), we should exploit more to refine the minimum.
    
    # Use a smooth decay of beta from high to low.
    # Total acquisition steps: 25 (30 total - 5 initial)
    # We want beta to decrease smoothly.
    
    # Smooth beta schedule:
    # Step 1: beta = 2.0 (high exploration)
    # Step 5: beta = 1.0
    # Step 10: beta = 0.5
    # Step 15: beta = 0.2
    # Step 20+: beta = 0.05 (near exploitation)
    
    if step <= 2:
        beta = 2.0
    elif step <= 5:
        beta = 1.0
    elif step <= 10:
        beta = 0.5
    elif step <= 15:
        beta = 0.2
    elif step <= 20:
        beta = 0.1
    else:
        beta = 0.05
        
    # UCB for minimization: -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
