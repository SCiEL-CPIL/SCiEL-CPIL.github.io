"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to 0 (shouldn't happen but safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Cost is at least 1, so this penalizes expensive evaluations
    acq = ei / (cost + 1e-9)
    
    # Ensure finite values
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
# EVOLVE-BLOCK-END
