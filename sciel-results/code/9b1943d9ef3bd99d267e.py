import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division issues
    sigma_safe = sigma + 1e-9
    
    # Calculate z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Add a small exploration term based on uncertainty
    # This helps escape local optima and explore cheap, uncertain regions
    exploration = 0.1 * sigma_safe
    
    # Cost-aware score: higher is better
    # We prefer high probability of improvement and high uncertainty, penalized by cost
    scores = (poi + exploration) / cost
    
    return scores
