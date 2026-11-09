"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Probability of Improvement (CA-POI)
    # POI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Avoid division by zero or very small sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z score
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    poi = norm.cdf(Z)
    
    # Ensure POI is non-negative (should be, but just in case of numerical noise)
    poi = np.maximum(poi, 0)
    
    # Cost-awareness: Divide by cost
    # Higher POI and lower cost is better
    scores = poi / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
