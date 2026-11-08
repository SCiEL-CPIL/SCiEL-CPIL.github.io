"""Rewritten program with cost-aware acquisition for Ackley 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, best observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return simple EI-based scores
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0) / np.maximum(cost, 1e-9)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    # Use a slightly modified formulation to handle edge cases
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
