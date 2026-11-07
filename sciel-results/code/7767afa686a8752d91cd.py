import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Gaussian Process Lower Confidence Bound (GP-LCB).
    mu: (N,) array of posterior means
    sigma: (N,) array of posterior std devs
    f_best: float, best observed value so far
    Returns: (N,) array of scores (higher = better)
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic beta: start with more exploration, decrease as we get closer to optimum
    # f_best is the minimum value found so far.
    # If f_best is large, we are far from 0, so explore more (higher beta).
    # If f_best is small, we are close to 0, so exploit more (lower beta).
    
    # Simple heuristic: beta = 2.0 / (1 + 0.1 * f_best)
    # If f_best = 25, beta = 2.0 / 3.5 = 0.57 (too low?)
    # If f_best = 0.1, beta = 2.0 / 1.01 = 1.98 (too high?)
    
    # Let's try: beta = 1.5
    # Or a fixed beta is safer.
    
    beta = 2.0
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize this, so we return - (mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
