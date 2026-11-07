import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Samples from the posterior distribution N(mu, sigma^2)
    and returns the negative of the sample (to maximize the minimum).
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to simple LCB
        return -mu + 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a sample from the posterior
    # We want to minimize f, so we look for the lowest sampled value.
    # To use "max score" interface, we return -sample.
    samples = np.random.normal(mu, sigma_safe)
    
    # Ensure finite
    scores = -samples
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
