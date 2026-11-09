"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: track elapsed time to avoid exceeding the 330s limit
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a cost penalty.
    """
    # Deadline guard: if we're running too long, return a safe fallback
    # that prioritizes exploration (high sigma) to avoid getting stuck.
    elapsed = time.time() - _start_time
    if elapsed > 330 - _DEADLINE_MARGIN:
        # Fallback: prioritize exploration (high sigma) to avoid getting stuck
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        scores = sigma_safe / cost_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting uncertain regions
    # when EI is very small (e.g., near the best point)
    # This helps escape local optima in the Levy function landscape
    exploration = sigma * 0.1
    
    # Combine EI and exploration
    combined = ei + exploration
    
    # Cost-aware acquisition: divide by cost
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    scores = combined / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
