"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function: Expected Improvement per unit cost."""
    # Clip sigma to avoid division by zero / numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For large negative z (mu >> f_best), EI ~ 0
    # For large positive z (mu << f_best), EI ~ f_best - mu
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Cost is 1 + mean(x scaled to [0,1]), so cost in [1, 2]
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
