import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) 
           + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    We return the EI score. Higher is better.
    """
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # EI calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is always non-negative
    ei = np.maximum(ei, 0.0)
    
    # Handle non-finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
