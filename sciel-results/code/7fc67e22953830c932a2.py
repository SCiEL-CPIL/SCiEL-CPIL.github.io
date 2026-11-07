"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on shekel 4d.
    Returns 2000 scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time to avoid timeout
    # We use a module-level timer that gets reset on first call
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    # If we're getting close to the 300s limit, return safe scores
    if elapsed > 290:
        return np.zeros_like(mu)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to avoid numerical overflow in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scoring: divide EI by cost^alpha
    # Using alpha=0.5 balances exploration and cost
    score = ei / (cost_safe ** 0.5)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
