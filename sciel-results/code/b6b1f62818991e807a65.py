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
    
    # Worst-Case Improvement (WCI) for minimization
    # Pessimistic estimate of the function value: mu + k * sigma
    # We only consider it an improvement if this pessimistic value is better than f_best
    # Score = max(0, f_best - (mu + k * sigma))
    k = 1.0  # Fixed exploration constant
    
    wci = f_best - mu - k * sigma_safe
    
    # Clip to zero to ensure non-negative scores
    wci = np.maximum(wci, 0.0)
    
    # Cost-aware normalization
    # Prefer high improvement per unit cost
    scores = wci / cost_safe
    
    # Fallback strategy:
    # If no point offers a guaranteed worst-case improvement (all scores ~ 0),
    # the GP is likely overconfident or f_best is very good.
    # In this case, switch to a pure exploration metric: prefer high uncertainty (sigma)
    # and low cost.
    if np.max(scores) < 1e-12:
        scores = sigma_safe / cost_safe
        
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
