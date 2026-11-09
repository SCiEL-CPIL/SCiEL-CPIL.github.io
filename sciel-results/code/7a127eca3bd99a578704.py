"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized acquisition combining Expected Improvement with a
    probability-of-improvement term weighted by uncertainty.
    
    Strategy: Use a blend of EI/cost and POI/cost, where POI adds exploration
    pressure. The POI term is scaled by sigma to balance exploitation (EI)
    and exploration (POI weighted by uncertainty).
    
    We use: score = (EI + beta * sigma * POI) / cost
    where beta controls the exploration-exploitation tradeoff.
    
    A small constant offset in the numerator prevents division by zero and
    ensures finite outputs even when cost is very small.
    """
    # Guard against zero/near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Blend: EI dominates when confident, POI*sigma adds exploration
    # beta=0.5 balances the two terms
    beta = 0.5
    numerator = ei + beta * sigma_safe * poi
    
    # Cost normalization with a small floor to avoid division by near-zero
    cost_safe = np.maximum(cost, 1e-6)
    
    score = numerator / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
