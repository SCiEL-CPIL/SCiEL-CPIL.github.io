import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    if time.time() - _start_time > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling
    # Draw a random standard normal
    r = np.random.normal(0, 1)
    
    # Score to maximize: -mu + r * sigma
    # This corresponds to minimizing mu - r * sigma (posterior sample)
    score = -mu + r * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
