"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost.
    Handles edge cases to ensure finite outputs.
    """
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using stable computation for large z
    # For z > 0, EI is positive. For z <= 0, EI is 0 (since we only care about improvement)
    # Actually, standard EI formula:
    # If z is very large positive, Phi(z) ~ 1, phi(z) ~ 0, EI ~ f_best - mu
    # If z is very large negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0
    
    # To avoid numerical issues, we can use the fact that EI is always non-negative
    # and for z < 0, the improvement is negligible.
    
    # Standard formula
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Cost is 1 + mean(x scaled to [0,1]), so cost is in [1, 2]
    # We want to maximize EI / cost
    score = ei / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
