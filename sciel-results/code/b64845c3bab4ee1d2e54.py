import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement per unit cost.
    
    EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    Cost-aware score = EI / cost
    """
    # Compute z-score
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: when mu < f_best, (f_best - mu) > 0, and EI > 0
    # When mu > f_best, EI can be small or zero
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Cost is in [1, 2], so this is a mild adjustment
    score = EI / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
