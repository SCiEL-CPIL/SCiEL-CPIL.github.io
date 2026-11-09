"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a geometric mean of normalized EI and inverse cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range using max EI
    max_ei = np.max(ei)
    if max_ei > 1e-9:
        ei_norm = ei / max_ei
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute inverse cost normalized
    inv_cost = 1.0 / cost
    max_inv_cost = np.max(inv_cost)
    min_inv_cost = np.min(inv_cost)
    
    # Normalize inverse cost to [0, 1]
    if max_inv_cost - min_inv_cost > 1e-9:
        inv_cost_norm = (inv_cost - min_inv_cost) / (max_inv_cost - min_inv_cost)
    else:
        inv_cost_norm = np.ones_like(inv_cost)
    
    # Geometric mean of normalized EI and normalized inverse cost
    # This balances exploitation (high EI) and efficiency (low cost)
    # Use log-space to avoid numerical issues
    eps = 1e-9
    log_ei = np.log(ei_norm + eps)
    log_inv_cost = np.log(inv_cost_norm + eps)
    
    # Geometric mean
    geo_mean = np.exp(0.5 * (log_ei + log_inv_cost))
    
    # Return finite scores
    scores = np.where(np.isfinite(geo_mean), geo_mean, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
