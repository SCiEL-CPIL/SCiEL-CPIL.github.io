"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_T0 = time.time()
_DEADLINE = 300.0

def _remaining_time():
    return _DEADLINE - (time.time() - _T0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if _remaining_time() < 1.0:
        # Return a safe fallback: uniform scores
        return np.ones(2000)
    
    try:
        # Ensure finite inputs
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Handle edge cases
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        
        # Standard Expected Improvement
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Add exploration bonus
        exploration_bonus = 0.1 * sigma
        
        # Combine
        combined = ei + exploration_bonus
        
        # Cost-aware: divide by cost
        scores = combined / cost
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
    
    except Exception:
        # Fallback: simple EI/cost
        try:
            sigma_safe = np.maximum(sigma, 1e-9)
            cost_safe = np.maximum(cost, 1e-9)
            z = (f_best - mu) / sigma_safe
            ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            scores = ei / cost_safe
            scores = np.where(np.isfinite(scores), scores, 0.0)
            return scores
        except Exception:
            return np.ones(2000)
# EVOLVE-BLOCK-END
