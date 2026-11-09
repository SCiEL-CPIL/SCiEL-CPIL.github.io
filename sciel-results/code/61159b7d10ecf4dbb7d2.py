import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Expected Improvement (EI) for minimization
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score: (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI / cost
    # Higher EI and lower cost lead to higher score
    score = EI / cost
    
    # Ensure finite scores
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
