import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # We want to minimize f, so improvement is when f < f_best
    # EI = E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    #
    # Standard EI formula for minimization:
    # Let z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    #
    # For points where sigma is very small, we need to handle carefully.
    # Higher EI means more promising to evaluate next.
    
    # Ensure sigma is at least a tiny positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # Phi(z) = CDF of standard normal
    # phi(z) = PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
