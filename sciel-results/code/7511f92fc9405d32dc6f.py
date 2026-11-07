"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    UCB-based cost-aware acquisition function.
    
    mu, sigma: arrays of shape (2000,) - GP mean and std in objective units
    f_best: float - lowest observed value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative costs
    cost = np.maximum(cost, 1e-6)
    
    # Standard deviation guard
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to minimize, so lower mu is better.
    # UCB = mu - beta * sigma (for minimization, we explore where sigma is high)
    # But we want a score where higher = better, so we negate:
    # score_raw = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Choose beta to balance exploration and exploitation
    # A fixed beta of 2.0 is a reasonable starting point
    beta = 2.0
    
    # Compute UCB-based score (higher is better for selection)
    ucb_score = -mu + beta * sigma
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    # Use linear cost penalty as suggested
    total_score = ucb_score / cost
    
    # Safety: ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
