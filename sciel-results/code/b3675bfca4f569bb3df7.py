import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement for minimization
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement for minimization
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
