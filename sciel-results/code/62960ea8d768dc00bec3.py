import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 290  # 10s safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Check time budget
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a simple fallback
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / np.sqrt(cost_safe)
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
