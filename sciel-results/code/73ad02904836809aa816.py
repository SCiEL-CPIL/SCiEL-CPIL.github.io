import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Probability of Improvement
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost normalization: use 1/cost for stronger penalty
    cost_safe = np.maximum(cost, 1e-10)
    scores = pi / cost_safe
    
    # Add small exploration bonus for high uncertainty
    exploration = np.log1p(sigma) * 0.1
    scores = scores + exploration
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
