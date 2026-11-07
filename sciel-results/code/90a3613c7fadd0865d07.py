"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_CALL_COUNT = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    global _START_TIME, _CALL_COUNT
    _CALL_COUNT += 1
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 330s limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 320:  # 10s safety margin
        # Return a conservative score that prefers lower cost
        safe_score = 1.0 / (cost + 1e-9)
        return np.where(np.isfinite(safe_score), safe_score, 0.0)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But also add a small exploration bonus based on uncertainty
    # to avoid getting stuck in exploitation-only mode
    
    # Hybrid: combine EI/cost with a small uncertainty bonus
    # The uncertainty bonus helps explore when EI is low
    uncertainty_bonus = sigma * 0.1  # Small weight on uncertainty
    
    score = ei / (cost + 1e-9) + uncertainty_bonus / (cost + 1e-9)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
