"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for the deadline guard
_start_time = time.time()
TIME_LIMIT = 295.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Maximizes: kappa * sigma - mu - lambda * cost
    This is equivalent to minimizing: mu - kappa * sigma + lambda * cost
    """
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback: simple exploitation with light cost penalty
        return -mu - 0.1 * cost

    # Parameters
    kappa = 1.5   # Exploration factor
    lam = 0.25    # Cost penalty factor
    
    # Calculate LCB score
    # We want to find points with low mu (good), high sigma (uncertain), and low cost.
    # Score = kappa * sigma - mu - lam * cost
    score = kappa * sigma - mu - lam * cost
    
    # Ensure finite output to satisfy contract
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
