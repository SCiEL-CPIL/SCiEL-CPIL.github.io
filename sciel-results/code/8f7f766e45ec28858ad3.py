import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # To avoid log(0) or negative values (EI should be non-negative, but numerically might dip)
    # Clip EI to a small positive value
    ei = np.maximum(ei, 1e-10)
    
    # Log Expected Improvement
    scores = np.log(ei)
    
    return scores
