import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus to prevent ignoring high-uncertainty regions
    # where EI might be small due to high mean
    exploration = 0.2 * sigma_safe
    
    score = (ei + exploration) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
