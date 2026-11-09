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
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement.
    
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
        # Fallback: Fast linear approximation of EI - penalty*cost + bonus*sigma
        # EI approx: max(0, f_best - mu) is a crude bound, but (f_best - mu + k*sigma) is better
        # Let's use a simple UCB-like score minus cost penalty
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        # f_best - mu is the potential improvement. Add exploration. Subtract cost penalty.
        return (f_best - mu + 2.0 * sigma_safe) - 0.1 * cost_safe

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
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty and Exploration bonus
    # Score = EI - lambda * cost + beta * sigma
    # lambda penalizes expensive evaluations
    # beta encourages exploration in high-uncertainty regions
    lambda_cost = 0.1
    beta_explore = 1.5
    
    score = ei - lambda_cost * cost_safe + beta_explore * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
