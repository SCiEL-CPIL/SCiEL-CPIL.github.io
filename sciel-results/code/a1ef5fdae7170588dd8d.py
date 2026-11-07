# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement normalized by cost, with adaptive exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of GP predictions in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation of GP predictions in objective units.
    f_best : float
        Best observed value so far (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 250.0  # 300s limit, keep 50s margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure no NaN or Inf inputs propagate
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e10, neginf=1.0)
    
    # Safety: ensure sigma is positive (tiny epsilon)
    sigma = np.maximum(sigma, 1e-9)
    
    # Safety: ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus based on uncertainty
    # Use a beta parameter that adapts based on the spread of sigma values
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    sigma_range = sigma_max - sigma_min
    
    # Adaptive beta: larger when uncertainty is high, smaller when converged
    if sigma_range > 1e-6:
        beta = 1.0 + 0.5 * (sigma / (sigma_range + 1e-9))
    else:
        beta = 1.0
    
    # Combine EI with exploration bonus
    acquisition_value = ei + beta * sigma
    
    # Cost-aware normalization
    # Use inverse cost weighting to favor cheaper evaluations
    # Normalize costs to [0, 1] range for better conditioning
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-9:
        cost_normalized = (cost - cost_min) / cost_range
    else:
        cost_normalized = np.zeros_like(cost)
    
    # Weighted cost penalty: cheaper points get higher scores
    # Use a power law to control the strength of cost awareness
    cost_weight = 1.0 / (1.0 + cost_normalized)
    
    scores = acquisition_value * cost_weight
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Safety check: if any non-finite values remain, replace with 0
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Deadline check
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
