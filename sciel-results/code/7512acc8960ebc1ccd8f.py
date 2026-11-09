import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Cost-aware score: POI per unit cost
    # We add a small epsilon to cost to avoid division by zero (though cost >= 1)
    scores = poi / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
