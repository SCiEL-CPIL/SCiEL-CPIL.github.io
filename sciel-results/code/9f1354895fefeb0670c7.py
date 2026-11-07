import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Variance Penalty
    # The variance is sigma^2. We subtract a small penalty to discourage
    # exploring regions with high uncertainty unless the EI is very high.
    # This helps in high-dimensional settings where GP uncertainty can be noisy.
    variance_penalty = 0.01 * sigma**2
    
    # Numerator: EI - Penalty
    numerator = ei - variance_penalty
    
    # Cost-aware score
    # We want higher scores for better candidates.
    # Cost is in the denominator, so cheaper points get higher scores for the same EI.
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = numerator / cost_safe
    
    # Handle NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
