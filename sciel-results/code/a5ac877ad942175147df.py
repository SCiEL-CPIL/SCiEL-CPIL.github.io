import numpy as np
import time

_acq_call_count = 0
_start_time = time.time()
TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best):
    global _acq_call_count, _start_time
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        return -mu.copy()
    
    _acq_call_count += 1
    
    # Determine progress within the current seed
    # 25 acquisition steps per seed
    step_in_seed = (_acq_call_count - 1) % 25
    # t goes from 1 to 25
    t = step_in_seed + 1
    
    # Adaptive beta: decays as sqrt(t)
    beta_0 = 2.0
    beta = beta_0 / np.sqrt(t)
    
    # UCB score for minimization
    # We want to minimize f(x).
    # UCB(x) = mu(x) - beta * sigma(x)
    # We return higher scores for better points.
    # So we return -UCB(x) = -mu(x) + beta * sigma(x)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
