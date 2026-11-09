"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    # Check time limit
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return a simple heuristic to finish quickly
        # Prefer low cost and low mu
        return -mu / cost

    # Lower Confidence Bound (LCB) with beta = 2.0
    # Minimization: we want low mu and high sigma
    # Score = (beta * sigma - mu) / cost
    beta = 2.0
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
