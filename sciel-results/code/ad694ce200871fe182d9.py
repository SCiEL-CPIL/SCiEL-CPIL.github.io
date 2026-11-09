import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # This is a structural change from UCB to EI, which is typically more robust
    # and often the standard baseline for GP-BO.
    
    # Handle zero or near-zero sigma to avoid numerical instability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the standardized improvement z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
