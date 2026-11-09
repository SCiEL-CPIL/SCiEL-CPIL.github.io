"""Cost-aware acquisition function for Bayesian optimization on 3-hump camel 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement divided by Cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = more desirable)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fast fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Expected Improvement (for minimization) ---
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # To avoid numerical issues with very large z (where Phi(z) -> 1 and phi(z) -> 0),
    # we can use a stable calculation, but scipy's norm.cdf and norm.pdf are generally stable.
    
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # For very negative z (mu >> f_best), EI is 0.
    # For very positive z (mu << f_best), EI approx (f_best - mu).
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Cost-Aware Score ---
    # Strict linear cost penalty: EI / Cost
    # This prioritizes candidates that offer high improvement per unit of cost.
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
