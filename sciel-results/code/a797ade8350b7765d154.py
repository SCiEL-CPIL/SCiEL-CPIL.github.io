import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    # We divide by cost to make it cost-aware.
    
    # Ensure sigma is non-negative and add small epsilon to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: higher is better
    # We want to maximize EI per unit cost
    score = ei / cost
    
    return score
