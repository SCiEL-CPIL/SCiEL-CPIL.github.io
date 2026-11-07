"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware Expected Improvement per unit cost (EIpUC) for minimization."""
    # EIpUC: maximize EI(x) / cost(x)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # This naturally balances exploration (high sigma) and exploitation (low mu)
    # while penalizing expensive evaluations.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # Use numerical stability for large |z|
    # For very negative z (mu >> f_best), EI ~ 0
    # For very positive z (mu << f_best), EI ~ f_best - mu
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Compute EIpUC: EI per unit cost
    # Cost is in [1, 2], so dividing by cost penalizes expensive points
    eipuc = ei / cost
    
    # Ensure finite scores, replace non-finite with very negative values
    eipuc = np.where(np.isfinite(eipuc), eipuc, -1e10)
    
    return eipuc
# EVOLVE-BLOCK-END
