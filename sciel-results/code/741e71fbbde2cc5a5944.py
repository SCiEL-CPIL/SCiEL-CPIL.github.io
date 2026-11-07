"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) acquisition function for minimization.
    # EI balances exploitation (low mu) and exploration (high sigma).
    # For minimization, improvement is (f_best - f(x)).
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle zero or near-zero sigma to avoid division by zero
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
