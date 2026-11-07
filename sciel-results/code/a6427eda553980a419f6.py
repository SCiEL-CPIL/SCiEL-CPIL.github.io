# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost
    with a tunable exploration-exploitation balance.
    
    The key insight: we want to maximize expected improvement per unit cost.
    EI naturally balances exploration (high sigma) and exploitation (low mu).
    Dividing by cost penalizes expensive evaluations.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardized distance: how many sigmas below f_best is mu?
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to avoid extreme values
    ei = np.clip(ei, 0, 1e6)
    
    # Cost-aware acquisition: EI per unit cost
    # This directly optimizes for improvement per dollar spent
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
