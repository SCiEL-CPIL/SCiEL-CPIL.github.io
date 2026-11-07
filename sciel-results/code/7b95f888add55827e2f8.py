import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative and not too small to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # Note: We want to minimize, so improvement is f_best - f(x)
    # E[max(0, f_best - f(x))]
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Replace non-finite values with 0 (or a very small negative number to deprioritize)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
