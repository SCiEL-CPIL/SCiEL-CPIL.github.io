import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    penalized by evaluation cost.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    DEADLINE = 290.0  # Safety margin before 300s kill

    # Convert inputs to numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-6)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    t_elapsed = time.time() - t_start
    if t_elapsed > DEADLINE:
        return scores
        
    return scores
