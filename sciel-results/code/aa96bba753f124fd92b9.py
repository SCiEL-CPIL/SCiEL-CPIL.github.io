import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Handle sigma close to 0 to avoid division by zero
    # We add a small epsilon to sigma to prevent numerical instability
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the Z-score
    # z = (f_best - mu) / sigma
    # If sigma is 0, z is effectively infinity if f_best > mu, else -infinity
    # We can compute z safely using np.where or just divide and handle later
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best < mu, EI is 0. The formula naturally yields negative values or close to 0,
    # but strictly EI is defined as max(0, ...).
    # Let's compute the raw EI value.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    return ei
