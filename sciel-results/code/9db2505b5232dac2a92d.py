"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300.0  # safety margin before 330s
    return _DEADLINE

def _time_left():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) with a calibrated exploration bonus.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - evaluation costs
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Check deadline: if we're close to the limit, return a fast fallback
    if _time_left() < 10.0:
        # Fast fallback: normalized negative mu (exploitation only)
        mu = np.asarray(mu, dtype=np.float64)
        mu_std = np.std(mu)
        if mu_std > 1e-9:
            return -mu / mu_std
        return np.zeros_like(mu)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative (should be, but numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / cost
    
    # Exploration bonus: proportional to sigma, normalized and scaled
    # This encourages exploration when EI is low
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Exploration weight: moderate to balance exploitation and exploration
    exploration_weight = 0.15
    exploration_bonus = exploration_weight * sigma_norm / cost
    
    # Combined score
    scores = ei_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
