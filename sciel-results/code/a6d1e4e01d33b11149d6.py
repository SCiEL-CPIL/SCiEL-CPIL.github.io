# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Expected Improvement (EI) for minimization.
    # EI estimates the expected reduction in the best observed value.
    # Higher EI means more likely to find a new global minimum.
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the normalized distance from the best found value
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    
    # Compute CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI scores
    # Note: If mu is much larger than f_best, z is large negative.
    # Phi(z) is small, (f_best - mu) is negative large.
    # The term sigma * phi(z) dominates if z is not too negative?
    # Actually, EI is always non-negative.
    scores = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
