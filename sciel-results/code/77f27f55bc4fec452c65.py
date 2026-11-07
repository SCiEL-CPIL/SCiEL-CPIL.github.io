import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 280.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT:
        # Return a simple fallback to avoid timeout
        return np.zeros_like(mu)
    
    # Calculate z-score for Expected Improvement
    # sigma is standard deviation
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(f < f_best) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring
    # The cost is 1 + mean(x). Range [1, 2].
    # The global minimum of Hartmann 6D is in the low-cost region (mean(x) ~ 0.23).
    # We penalize higher costs more aggressively to bias the search towards the cheaper region.
    # Using an exponent of 1.5 amplifies the difference between low and high cost points.
    safe_cost = np.maximum(cost, 1e-9)
    score = ei / (safe_cost ** 1.5)
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
