import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
