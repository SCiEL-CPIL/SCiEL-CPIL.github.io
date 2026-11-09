import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Handle edge cases where sigma is very small
    sigma = np.maximum(sigma, 1e-8)
    
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # E[max(f_best - f, 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return ei
