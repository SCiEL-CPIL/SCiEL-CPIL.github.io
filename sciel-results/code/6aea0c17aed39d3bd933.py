import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: small penalty for cost
    # Cost is in [1, 2]. EI scale depends on f_best.
    # Use a relative penalty?
    # Let's try a fixed small penalty.
    cost_penalty = 0.05 * cost
    
    return ei - cost_penalty
