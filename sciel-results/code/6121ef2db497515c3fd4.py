import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Effective Expected Improvement (CE-EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # Score = EI(x) / cost(x)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF of standard normal
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # If mu < f_best, (f_best - mu) is positive.
    # If mu > f_best, (f_best - mu) is negative, but Phi(z) is small, making EI small.
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: Expected Improvement per unit cost
    # Higher score is better
    safe_cost = cost + 1e-9
    scores = EI / safe_cost
    
    return scores
