"""Rewritten program: cost-aware acquisition using exponential cost penalty on Probability of Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses Probability of Improvement with an exponential cost penalty:
        score = P(improvement) * exp(-beta * cost)
    
    This creates a sharp preference for cheap, likely-improving points,
    which can escape local optima caused by the GP's high-variance regions
    that EI favors but which are often expensive or unreliable.
    """
    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Exponential cost penalty
    # cost is in range [1, 2] (1 + mean of x scaled to [0,1])
    # Use beta to control how strongly we penalize expensive points
    # A larger beta means we more strongly prefer cheap points
    beta = 2.0
    cost_penalty = np.exp(-beta * cost)
    
    score = pi * cost_penalty
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
