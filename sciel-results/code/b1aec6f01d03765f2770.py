"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI is a deterministic acquisition function that measures the expected improvement
    over the current best value f_best. It balances exploration and exploitation
    analytically without requiring random sampling, which reduces variance in the
    selection process and can lead to more consistent progress.
    
    For minimization, the improvement is defined as max(0, f_best - y).
    The expected improvement for a Gaussian posterior y ~ N(mu, sigma^2) is:
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    We return the EI value as the score (higher is better).
    """
    # Ensure sigma is strictly positive to avoid division by zero or log(0) issues
    # The problem states sigma is standard deviation, but numerically it might be 0 or negative due to floating point
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute the normalized difference
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If sigma is very small, z is large. 
    # For z -> infinity, Phi(z) -> 1, phi(z) -> 0. EI -> f_best - mu.
    # For z -> -infinity, Phi(z) -> 0, phi(z) -> 0. EI -> 0.
    
    # Using scipy.stats.norm for stability
    # norm.cdf(z) and norm.pdf(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finiteness
    # If mu is inf or nan, handle it. But mu/sigma are from GP, should be finite.
    # If f_best is inf? Unlikely, but safe to check.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
