import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Effective Expected Improvement (CEI) for minimization
    # EI = E[max(0, f_best - f(x))]
    # Score = EI / cost
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, high probability of improvement
    # If mu > f_best, z < 0, low probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF of standard normal
    # Using scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # (f_best - mu) * P(f(x) < f_best) + sigma * pdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    scores = ei / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
