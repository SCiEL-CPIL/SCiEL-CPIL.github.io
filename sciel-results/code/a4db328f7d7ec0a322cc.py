"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and an exploration term (sigma).
    
    Score = (EI + lambda * sigma) / cost
    Higher score means higher priority for evaluation.
    """
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Exploration term
    # Adding a portion of the standard deviation encourages exploration
    # of areas with high uncertainty, which is crucial in early stages
    # or when the GP is not well-calibrated.
    lambda_exp = 0.5
    
    # Combine EI and exploration, then normalize by cost
    # Cost is >= 1.0
    scores = (ei + lambda_exp * sigma) / cost
    
    return scores
# EVOLVE-BLOCK-END
