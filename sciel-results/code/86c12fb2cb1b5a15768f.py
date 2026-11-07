import numpy as np
import time

_start_time = None
_DEADLINE_SECONDS = 325.0

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if _elapsed() > _DEADLINE_SECONDS:
        # Fallback to simple negative mean if running out of time
        return -np.asarray(mu, dtype=np.float64).copy()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization
    # Score = - (mu - beta * sigma) = -mu + beta * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
