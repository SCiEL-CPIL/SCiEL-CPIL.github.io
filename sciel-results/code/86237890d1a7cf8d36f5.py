import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # Fallback: Fixed LCB
        return -mu - 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI) for minimization
    # We want P(f(x) < f_best)
    # Given f(x) ~ N(mu(x), sigma(x)^2)
    # P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    # To avoid division by zero or very small numbers, we use sigma_safe
    z = (f_best - mu) / sigma_safe
    
    # Compute the CDF of the standard normal distribution
    scores = norm.cdf(z)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
