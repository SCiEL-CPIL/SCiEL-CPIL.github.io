"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level timer for the deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) logic for minimization.
    Since we want to minimize f, we maximize the score.
    Score = -mu + beta * sigma
    This balances exploitation (low mu) and exploration (high sigma).
    """
    # Deadline guard: Check if we are close to the 330s limit.
    # If so, return a safe, exploitation-heavy score to ensure we finish.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Pure exploitation: pick lowest mean
        return -np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # UCB parameter
    beta = 1.0
    
    # Compute scores
    # We want high scores for low mu and high sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
