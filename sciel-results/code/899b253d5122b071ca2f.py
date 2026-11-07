import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    # where Z = (f_best - mu(x)) / sigma(x)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi is norm.cdf, phi is norm.pdf
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
