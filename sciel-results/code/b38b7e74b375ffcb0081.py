"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by cost.
    
    EI balances exploration and exploitation better than PI.
    We use:
        EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        where z = (f_best - mu) / sigma
    
    The final score is EI / cost, encouraging efficient improvements.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * P(f < f_best) + sigma * pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI should be non-negative, but numerical errors might make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost
    # Cost is in range [1, 2]
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
