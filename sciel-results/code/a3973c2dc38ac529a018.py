import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best):
    # Explicit deadline guard
    if time.time() - _start_time > _TIMEOUT:
        # If we are close to timeout, return a simple heuristic to ensure we finish fast
        # Return -mu (greedy) or just zeros to break the loop quickly if possible?
        # The evaluator picks the max score. If we return zeros, it picks the first one?
        # Better to return -mu to at least try to minimize.
        return -mu
        
    # Thompson Sampling
    # Draw random samples from the posterior
    z = np.random.randn(mu.shape[0])
    f_sample = mu + sigma * z
    
    # We want to minimize f_sample.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    return -f_sample
