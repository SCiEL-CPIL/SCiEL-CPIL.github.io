"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (CA-EI)
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z score
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # phi(Z) is the standard normal PDF
    # Phi(Z) is the standard normal CDF
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Cost-awareness: Divide by cost
    # Higher EI and lower cost is better
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
