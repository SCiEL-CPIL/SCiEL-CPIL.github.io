import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    Returns a score where higher values indicate more promising points.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values (norm.cdf/pdf can sometimes return 0 or inf for extreme inputs, 
    # though with safe sigma it should be fine)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
