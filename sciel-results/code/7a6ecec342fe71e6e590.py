import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Scaling factor for exploration
    # Standard EI is k=1. 
    # Higher k encourages more exploration.
    # EHVI with adaptive ref was effectively k > 1.
    k = 2.5
    
    # Scaled sigma
    sigma_scaled = k * sigma
    
    # z-score with scaled sigma
    z = (f_best - mu) / sigma_scaled
    
    # Expected Improvement with scaled sigma
    # E[max(0, f_best - Y)] where Y ~ N(mu, sigma_scaled^2)
    ei = (f_best - mu) * norm.cdf(z) + sigma_scaled * norm.pdf(z)
    
    scores = ei
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
