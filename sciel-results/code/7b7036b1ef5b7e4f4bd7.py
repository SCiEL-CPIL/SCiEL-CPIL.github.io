import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate z-score
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0)
    
    # Use Log-EI to compress the dynamic range and potentially encourage more balanced exploration
    # Adding a small epsilon to avoid log(0)
    scores = np.log(ei + 1e-9)
    
    return scores
