import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # Handle sigma == 0
    z = np.zeros_like(mu)
    mask = sigma > 1e-8
    z[mask] = (f_best - mu[mask]) / sigma[mask]
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) < f_best, EI > 0.
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score
    # We want to maximize EI per unit cost
    score = ei / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
