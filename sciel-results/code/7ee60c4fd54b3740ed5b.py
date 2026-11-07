import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost penalty exponent alpha
    # Alpha starts near 1.0 (strong cost penalty) and decays to 0.5 (weaker cost penalty)
    # as f_best improves (decreases towards 0).
    # Using a smooth transition based on f_best.
    # If f_best is large, 1/(f_best+1) is small, alpha ~ 1.
    # If f_best is 0, 1/(0+1) is 1, alpha ~ 0.5.
    alpha = 1.0 - 0.5 / (f_best + 1.0)
    
    # Ensure alpha is within reasonable bounds
    alpha = np.clip(alpha, 0.1, 1.5)
    
    # Calculate cost-aware score
    # Score = EI / cost^alpha
    # cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / (cost_safe ** alpha)
    
    # Return finite scores
    return score
