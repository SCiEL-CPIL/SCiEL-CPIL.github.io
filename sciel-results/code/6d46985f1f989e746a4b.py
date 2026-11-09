"""Cost-aware acquisition for Bayesian optimization on three-hump camel 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # safety margin before 330s kill

def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a UCB exploration bonus, normalized by evaluation cost.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions in objective units
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best (lowest) observed value
        cost: array of shape (2000,), evaluation costs
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        # Fallback: simple UCB-like score that's cheap to compute
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        return (f_best - mu + 2.0 * sigma_safe) / cost_safe
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues with very large/small values
    z_clipped = np.clip(z, -10, 10)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration bonus: beta * sigma
    # Use a moderate exploration coefficient
    beta = 1.5
    ucb_bonus = beta * sigma_safe
    
    # Combine: weighted sum of EI and UCB bonus, normalized by cost
    # The weighting balances exploitation (EI) and exploration (UCB)
    score = (ei + ucb_bonus) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
