import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization with cost awareness
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma)
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Then divide by cost to get cost-aware score
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate standardized improvement
    # For minimization, improvement is f_best - f(x)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Cost-aware: divide by cost
    cost_safe = np.where(cost > 1e-9, cost, 1.0)
    scores = ei / cost_safe
    
    # Replace any non-finite values with a large negative number
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
