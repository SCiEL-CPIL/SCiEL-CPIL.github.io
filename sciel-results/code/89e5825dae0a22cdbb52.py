import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for Probability of Improvement
    # We want P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher is better
    # Divide by cost to prefer cheaper points
    # Ensure cost is not zero (min cost is 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = pi / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
