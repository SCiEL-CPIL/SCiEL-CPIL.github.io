"""Rewritten program using Cost-Adjusted Expected Improvement (CEI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CEI).
    
    This approach calculates the Expected Improvement (EI) for each candidate point
    and divides it by the evaluation cost. This directly optimizes the expected
    improvement per unit of cost, which is a standard and effective strategy for
    cost-aware Bayesian optimization.
    
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    Score = EI / cost
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # For minimization, EI is the expected amount by which f(x) is lower than f_best
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware weighting
    # Cost is in [1, 2]. We divide by cost to penalize expensive points.
    # This creates a direct trade-off: a point must have significantly higher EI
    # to justify a higher cost.
    score = ei / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
