"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # 330s wall-clock limit, use 300s as safety margin
        _DEADLINE = time.time() + 300.0
    return _DEADLINE

def _time_remaining():
    return _get_deadline() - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure we have enough time
    if _time_remaining() < 1.0:
        # Emergency: return simple cost-adjusted improvement
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring: EI per unit cost
    # Use a slightly more robust cost adjustment
    cost_safe = np.maximum(cost, 1e-6)
    
    # Primary score: EI / cost
    scores = ei / cost_safe
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps when EI is small but sigma is large (unexplored regions)
    # Weighted by inverse cost to maintain cost-awareness
    exploration_bonus = sigma / cost_safe
    scores = scores + 0.1 * exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
