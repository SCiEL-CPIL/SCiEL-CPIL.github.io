import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 330
_SAFETY_MARGIN = 10

def acquisition(mu, sigma, f_best, cost):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback to simple score if we are running out of time
        # This should rarely happen as computation is fast
        return np.ones_like(mu) / cost
    
    # Lower Confidence Bound for minimization
    # We want to minimize f(x), so we look for low mu and high sigma.
    # LCB = mu - beta * sigma
    # We want to maximize the score, so score = -LCB / cost = (beta * sigma - mu) / cost
    
    beta = 2.5
    score = (beta * sigma - mu) / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
