"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # For minimization:
    # Improvement I(x) = max(0, f_best - f(x))
    # f(x) ~ N(mu, sigma^2)
    # E[I(x)] = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # We use scipy.stats.norm for accurate CDF and PDF
    # Phi(z) is the CDF
    # phi(z) is the PDF
    
    # Handle potential numerical issues where z might be very large or small
    # norm.cdf and norm.pdf handle these well, but we ensure finite output
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition. 
    # In regions where mu >> f_best, z is very negative, cdf(z) ~ 0, pdf(z) ~ 0, so ei ~ 0.
    # In regions where mu << f_best, z is very positive, cdf(z) ~ 1, so ei ~ f_best - mu.
    
    # We return EI as the score. Higher EI means higher expected improvement.
    # This directly aligns with the goal of minimizing the function.
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
