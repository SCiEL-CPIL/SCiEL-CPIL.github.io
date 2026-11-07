import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle case where sigma is zero or very small to avoid division by zero
    # If sigma is 0, improvement is max(f_best - mu, 0)
    # Using a small epsilon for stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the normalized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf is Phi, norm.pdf is phi
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite and non-negative (EI is always >= 0)
    # Clip very small negative values due to numerical precision to 0
    ei = np.maximum(ei, 0.0)
    
    # Replace any potential NaNs or Infs with 0
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
