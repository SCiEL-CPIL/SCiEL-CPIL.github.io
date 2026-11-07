import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300  # 300s limit, 30s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a weighted-sum Pareto scalarization.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple, fast score to avoid further computation
        return -mu

    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    # Numerical stability for cdf and pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term
    beta = 0.2
    exploration = beta * sigma
    
    # Combine EI and exploration
    improvement = ei + exploration
    
    # Normalize improvement and cost for weighted-sum scalarization
    # This decouples the improvement magnitude from the cost scaling
    imp_max = np.max(improvement)
    imp_min = np.min(improvement)
    imp_range = imp_max - imp_min
    if imp_range > 1e-12:
        imp_norm = (improvement - imp_min) / imp_range
    else:
        imp_norm = np.zeros_like(improvement)
    
    cost_max = np.max(cost)
    cost_min = np.min(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Invert cost so that lower cost is better
    cost_inv_norm = 1.0 - cost_norm
    
    # Weighted-sum scalarization
    w_imp = 0.7
    w_cost = 0.3
    scores = w_imp * imp_norm + w_cost * cost_inv_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
