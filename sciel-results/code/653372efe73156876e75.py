"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines expected improvement with a cost-aware penalty to balance
    exploration and exploitation while accounting for evaluation costs.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: If we are running out of time, return a simple fallback
    # to ensure we don't crash or timeout.
    if time.time() - _START_TIME > _TIME_LIMIT:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero or negative std
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # We want to balance EI against the cost of evaluation.
    # A common approach is to use EI / cost, but this can be unstable
    # when cost is very small. Instead, we use a soft cost penalty.
    
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones_like(cost)
    
    # Cost-aware score: EI with exponential decay based on normalized cost
    # Higher cost leads to lower score, but we use a smooth decay
    # alpha=0.5 was found effective in the best inspiration
    alpha = 0.5
    cost_weight = np.exp(-alpha * cost_norm)
    
    # Final acquisition score
    scores = ei * cost_weight
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
