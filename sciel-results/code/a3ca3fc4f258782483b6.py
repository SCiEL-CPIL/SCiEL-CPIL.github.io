import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Clip sigma to avoid division by zero or numerical instability
    sigma = np.maximum(sigma, 1e-9)
    
    # Use known global minimum 0 as a floor for the reference value
    # This makes the acquisition function less greedy when f_best > 0,
    # potentially aiding escape from local minima.
    f_ref = np.minimum(f_best, 0.0)
    
    z = (f_ref - mu) / sigma
    
    # Expected Improvement
    # EI = (f_ref - mu) * Phi(z) + sigma * phi(z)
    ei = (f_ref - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # Ensure cost is not zero (it's >= 1)
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
