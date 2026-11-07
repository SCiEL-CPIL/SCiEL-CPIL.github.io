"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Strategy:
    EI is a deterministic acquisition function that measures the expected improvement
    over the current best value f_best. It balances exploration and exploitation by
    considering both the predicted mean (mu) and the uncertainty (sigma).
    
    For minimization:
    Improvement = max(0, f_best - y)
    where y ~ N(mu, sigma^2).
    
    The analytical solution for EI is:
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    where Phi is the CDF of the standard normal distribution and phi is the PDF.
    
    We return the EI values as scores. Higher EI means higher expected improvement,
    which corresponds to a better candidate to evaluate next.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # In GP posterior, sigma should be > 0, but numerical issues can occur
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the improvement potential: f_best - mu
    # If mu < f_best, this is positive (potential improvement)
    # If mu > f_best, this is negative (unlikely to improve)
    imp = f_best - mu
    
    # Calculate the z-score for the standard normal distribution
    z = imp / sigma_safe
    
    # Calculate the standard normal CDF and PDF
    # norm.cdf(z) = Phi(z)
    # norm.pdf(z) = phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Compute Expected Improvement
    # EI = imp * Phi(z) + sigma * phi(z)
    ei = imp * cdf_z + sigma_safe * pdf_z
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative. Higher is better.
    # The task requires "higher meaning evaluate next", which matches EI directly.
    return ei
# EVOLVE-BLOCK-END
