import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: higher score for lower cost
    scores = ei / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
