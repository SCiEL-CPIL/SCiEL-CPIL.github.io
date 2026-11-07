import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: Mild penalty for high cost
    # cost is in [1, 2]
    # sqrt(cost) is in [1, 1.414]
    # Dividing by sqrt(cost) reduces score for expensive points by up to ~30%
    score = ei / np.sqrt(cost)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
