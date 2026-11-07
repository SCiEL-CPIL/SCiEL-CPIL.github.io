import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate adaptive weights for UCB and EI
    diff = max(f_best - 3.0, 0.0)
    
    # Weight for UCB: starts high, decays exponentially
    # weight_ucb = 1.0 when diff=0, decays to 0 as diff increases? No, opposite.
    # We want MORE UCB when diff is LARGE (early).
    # So weight_ucb should be HIGH when diff is LARGE.
    # weight_ucb = 1 - np.exp(-0.1 * diff)?
    # If diff=0, weight_ucb=0.
    # If diff=10, weight_ucb=0.63.
    # If diff=30, weight_ucb=0.95.
    
    weight_ucb = 1.0 - np.exp(-0.1 * diff)
    weight_ei = 1.0 - weight_ucb
    
    # UCB score
    beta = 2.0
    scores_ucb = -mu + beta * sigma
    
    # EI score
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    scores_ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine
    scores = weight_ucb * scores_ucb + weight_ei * scores_ei
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
