"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF of standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    return ei
# EVOLVE-BLOCK-END
