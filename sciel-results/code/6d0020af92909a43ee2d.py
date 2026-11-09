"""Rewritten program using a cost-aware Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    For minimization, PI measures the probability that the function value at x
    is lower than the current best f_best.
    
    PI(x) = Phi((f_best - mu) / sigma)
    
    We divide by cost to get cost-aware PI (cPI).
    Higher scores indicate better candidates to evaluate.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate the standardized improvement u
    # u = (f_best - mu) / sigma
    u = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    # PI = Phi(u)
    pi_vals = norm.cdf(u)
    
    # Cost-aware score: PI per unit cost
    # cost is given, and is >= 1.0
    scores = pi_vals / cost
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
