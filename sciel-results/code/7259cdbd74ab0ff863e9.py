import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Strategy
    # Sample a function from the posterior distribution
    # f_sample(x) ~ N(mu(x), sigma(x)^2)
    # We approximate this by sampling independent noise for each candidate
    z = np.random.randn(mu.shape[0])
    
    # Sampled values
    sampled_f = mu - sigma_safe * z
    
    # We want to minimize sampled_f / cost
    # So we maximize -sampled_f / cost
    scores = -sampled_f / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
