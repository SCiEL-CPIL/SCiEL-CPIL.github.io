import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    Phi_z = norm.cdf(z)
    Phi_z = np.where(np.isfinite(Phi_z), Phi_z, 0.0)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Higher score is better
    scores = ei / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
