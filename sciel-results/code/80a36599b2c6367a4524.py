import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
