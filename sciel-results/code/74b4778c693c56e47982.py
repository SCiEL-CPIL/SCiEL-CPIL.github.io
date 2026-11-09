"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Pareto front-based selection: computes non-dominated set with respect to
    maximizing Expected Improvement (EI) and minimizing acquisition cost,
    then scores candidates based on their rank within this front.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float
    cost: array of shape (2000,)
    
    Returns:
    score: array of shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for sigma and cost
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (for minimization)
    # EI = E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware acquisition: Expected Improvement per unit cost
    # This penalizes expensive evaluations while rewarding high expected gain.
    # Dividing by cost (range [1, 2]) effectively normalizes the gain.
    score = ei / cost_safe
    
    # Sanitize
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
