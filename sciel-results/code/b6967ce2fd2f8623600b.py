import numpy as np
import time

# Global state for deadline tracking
_start_time = time.time()
_TIMEOUT = 300.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    # If we are close to the timeout, return a fast, simple score
    # to ensure we don't hang.
    if time.time() - _start_time > _TIMEOUT - 5.0:
        # Greedy strategy: minimize mu
        # Return negative mu so that the minimum mu (best) has the highest score
        return -mu

    # Adaptive UCB
    # Exploration coefficient beta adapts to the current best value.
    # Higher f_best (worse solution) -> higher beta (more exploration).
    # Lower f_best (better solution) -> lower beta (more exploitation).
    
    # f_best is the minimum observed value.
    # We add a small epsilon to prevent beta from becoming 0 if f_best is very close to 0.
    eps = 1e-3
    beta = 1.5 * (f_best + eps)
    
    # Calculate UCB scores
    # Higher score means more likely to be evaluated
    scores = mu + beta * sigma
    
    return scores
