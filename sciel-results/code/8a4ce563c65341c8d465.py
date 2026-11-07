import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to a simple, fast score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return pi / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: if mu >= f_best, EI should be 0. 
    # The formula above yields a small positive number due to numerical precision, 
    # but (f_best - mu) is negative, so it might be negative. 
    # We should clip EI to be non-negative.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Weighted EI
    score = ei / cost_safe
    
    return score
