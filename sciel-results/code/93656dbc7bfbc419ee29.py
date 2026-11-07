"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 330s limit, use 300s as safety margin
    if time.time() - _bo_start_time > 300:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # The acquisition function for minimization is typically:
    # q(x) = mu(x) - beta * sigma(x)
    # We want to MAXIMIZE the score returned to the caller, so we return:
    # score(x) = -q(x) = -mu(x) + beta * sigma(x)
    
    # Beta controls the exploration/exploitation trade-off.
    # Higher beta -> more exploration.
    # For Ackley 2D with small budget, we choose a moderate exploration factor.
    beta = 2.0
    
    # Calculate score
    score = -mu + beta * sigma
    
    # Ensure finite values
    score[~np.isfinite(score)] = 0.0
    
    return score
# EVOLVE-BLOCK-END
