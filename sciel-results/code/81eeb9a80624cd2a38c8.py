import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement (EI)
    # delta = f_best - mu
    # z = delta / sigma
    # EI = delta * Phi(z) + sigma * phi(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Standard EI
    ei = np.where(delta > 0, delta * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
    
    # Cost-aware scoring
    # Use square root of cost to soften the penalty for high-cost regions
    # This allows the algorithm to explore more expensive areas if the expected improvement is high enough
    cost_soft = np.sqrt(cost)
    
    scores = ei / cost_soft
    
    return scores
