"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to pure exploitation (lowest mean)
        # Score = -mu
        return -mu

    # Use Upper Confidence Bound (UCB) for minimization
    # We want to select the candidate with the lowest Lower Confidence Bound (LCB)
    # LCB(x) = mu(x) - beta * sigma(x)
    # To make "higher score" mean "better candidate", we return -LCB(x)
    # Score(x) = -mu(x) + beta * sigma(x)
    
    # Beta parameter tuning
    # For 30 evaluations in 2D, beta=2.0 is a standard choice.
    # This corresponds to the 2.5th percentile of the posterior.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
