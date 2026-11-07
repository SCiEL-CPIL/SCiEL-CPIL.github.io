import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu.copy()
    
    # Thompson Sampling
    # Sample a single standard normal random variable
    z = np.random.normal(0, 1)
    
    # The sampled function value is mu + sigma * z
    # We want to minimize this, so we return the negative as the score
    # Higher score = Lower sampled value = Better candidate
    scores = -(mu + sigma * z)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
