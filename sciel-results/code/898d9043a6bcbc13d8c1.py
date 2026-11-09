"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: track elapsed time to ensure we finish within budget
    start_time = time.time()
    
    # Upper Confidence Bound (UCB) for minimization:
    # Score = -mu + beta * sigma
    # Higher score means more promising (lower expected value with uncertainty bonus)
    # This is a structural change from EI to UCB, which uses a different
    # exploration/exploitation trade-off formulation.
    
    # Compute the UCB score
    # beta controls the exploration-exploitation balance
    # For GP-BO with standardized targets, beta around 1.0-2.0 is typical
    beta = 1.5
    
    # UCB score: we want high score for low mu and high sigma
    score = -mu + beta * sigma
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check: if we're taking too long, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
