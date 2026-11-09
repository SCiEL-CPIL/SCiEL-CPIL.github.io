import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) for minimization.
    """
    # Prevent division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
