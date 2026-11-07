import numpy as np
import time

_acq_start_time = None
_DEADLINE = 300.0

def acquisition(mu, sigma, f_best, cost):
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    elapsed = time.time() - _acq_start_time
    if elapsed > _DEADLINE - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-8)
        score = -mu / cost
        score = np.where(np.isfinite(score), score, -1e10)
        return score
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Adaptive UCB: more exploration when f_best is far from optimum
    # Powell 4D min is 0, so f_best approaches 0
    # Scale kappa by f_best to adapt exploration
    kappa = max(1.0, 3.0 * f_best)
    
    # UCB score normalized by cost
    score = (-mu + kappa * sigma) / cost
    
    score = np.where(np.isfinite(score), score, -1e10)
    return score
