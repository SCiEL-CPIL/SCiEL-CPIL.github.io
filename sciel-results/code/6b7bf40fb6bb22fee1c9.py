import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 300.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Samples from the posterior distribution and returns the negative sampled value.
    The evaluator picks the highest score, which corresponds to the lowest sampled value.
    """
    # Check time
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple exploitation: return -mu (favor low mean)
        # Or just return zeros?
        # Returning -mu is a safe fallback.
        return -mu
    
    # Draw samples
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # Return negative samples so that lower function values yield higher scores
    scores = -samples
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
