import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure cost is positive and finite
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure f_best is finite
    f_best = np.where(np.isfinite(f_best), f_best, 0.0)
    
    # Compute z
    z = (f_best - mu) / sigma_safe
    
    # Clamp z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement for minimization
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Small exploration bonus
    exploration_bonus = 0.01 * sigma_safe
    
    # Cost-aware score
    score = (ei + exploration_bonus) / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
