import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: use a softer penalty to balance exploration vs cost
    # Using cost^0.5 instead of cost makes the penalty less severe,
    # allowing more exploration in early stages while still favoring cheaper points
    return ei / np.sqrt(cost)
