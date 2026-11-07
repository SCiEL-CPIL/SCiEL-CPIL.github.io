"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,)
        sigma: array of shape (2000,)
        f_best: scalar, lowest observed value
        cost: array of shape (2000,)
        
    Returns:
        array of shape (2000,) with scores (higher is better)
    """
    # Deadline guard: if we are close to the 330s limit, use a fast heuristic
    elapsed = time.time() - _start_time
    deadline = 300  # Safety margin
    
    if elapsed > deadline:
        # Fast path: prioritize low mu (likely good) and low cost
        # Avoid division by zero
        cost_safe = np.maximum(cost, 1e-9)
        # Use -mu / cost to favor low function value and low cost
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware score: Expected Improvement per unit cost
    # This directly optimizes the budget constraint
    score = ei / cost_safe
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
