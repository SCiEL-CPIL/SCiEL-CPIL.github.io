import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a safe default
    if time.time() > _DEADLINE - 5.0:
        # Return a simple score that's still valid
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0)
        return ei / cost
    
    # Ensure sigma is non-negative and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical underflow/overflow
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: use 1/sqrt(cost) instead of 1/cost
    # This reduces the bias towards the low-cost corner, allowing the
    # search to better locate minima in expensive regions
    score = ei / np.sqrt(cost)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
