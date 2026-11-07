"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 330s.
# We track time to ensure we return before the hard limit if computation were heavy.
# Here computation is O(N) and fast, but we comply with the directive.
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple heuristic if we are running out of time
        # This is unlikely for this O(N) operation
        return -mu + sigma

    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    # beta controls exploration. 
    # For 2D BO with ~30 samples, beta=2.0 is a reasonable exploration boost 
    # compared to EI's implicit ~0.4.
    
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
