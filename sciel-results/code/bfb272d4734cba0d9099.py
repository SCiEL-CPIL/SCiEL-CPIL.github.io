import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Expected Improvement (EI)
    # We want to minimize f.
    # Improvement I(x) = max(0, f_best - f(x))
    # Assuming f(x) ~ N(mu, sigma^2)
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    scores = ei / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
