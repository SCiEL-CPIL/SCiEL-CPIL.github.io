import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is theoretically non-negative, but numerical errors can make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Use Log-Expected Improvement for robustness
    # This reduces the influence of extremely high EI values (high uncertainty regions)
    # and encourages a more balanced exploration-exploitation trade-off.
    scores = np.log(ei + 1e-9)
    
    return scores
