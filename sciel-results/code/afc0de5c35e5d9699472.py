"""Direct acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Safety: ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Use a very small alpha to target the lower tail of the posterior
    # This approximates the minimum of many samples
    alpha = 1e-4
    z_alpha = norm.ppf(alpha)
    
    # Calculate the lower quantile of the posterior
    # q = mu + sigma * z_alpha
    # We want to minimize q, so we maximize -q
    scores = -(mu + sigma * z_alpha)
    
    # Check time
    if time.time() > deadline:
        return np.zeros_like(mu)
        
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
