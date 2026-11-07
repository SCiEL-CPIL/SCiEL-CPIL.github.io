"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization.
    # EI(x) = E[max(0, f_best - f(x))]
    # Assuming f(x) ~ N(mu, sigma^2)
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    from scipy.stats import norm
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite values
    return np.where(np.isfinite(ei), ei, 0.0)
# EVOLVE-BLOCK-END
