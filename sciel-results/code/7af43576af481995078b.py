import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, CDF > 0.5
    # If mu > f_best, z < 0, CDF < 0.5
    z = (f_best - mu) / sigma
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher is better
    # We want high PI and low cost
    score = pi / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
