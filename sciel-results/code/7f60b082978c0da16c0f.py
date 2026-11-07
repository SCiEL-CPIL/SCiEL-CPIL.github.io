"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _deadline_ok():
    return time.time() < _DEADLINE

def acquisition(mu, sigma, f_best):
    # Check deadline
    if not _deadline_ok():
        # Return a safe default (e.g., UCB with beta=0, which is just -mu)
        # Or just return -mu to prioritize exploitation if time is up
        return -mu

    # Structural Change: Upper Confidence Bound (UCB)
    # For minimization, we want to maximize the score.
    # Low mu is good (exploitation). High sigma is good (exploration).
    # Score = -mu + beta * sigma
    
    # Dynamic beta based on number of evaluations? 
    # We don't know the step number directly, but we can infer from f_best convergence or just use a fixed value.
    # A fixed beta of 2.5 is a standard choice for 2D BO.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
