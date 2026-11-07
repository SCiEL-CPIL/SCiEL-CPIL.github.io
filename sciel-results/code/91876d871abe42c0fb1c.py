import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Add a small epsilon to sigma to avoid division by zero
    sigma_safe = sigma + 1e-8
    
    Z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: higher is better
    # We want to maximize EI per unit cost
    score = ei / cost
    
    return score
