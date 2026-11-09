import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (LCB) for minimization, inverted for maximization score.
    # Score = -mu + beta * sigma
    # This favors low mean values and high uncertainty.
    
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return -mu

    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
