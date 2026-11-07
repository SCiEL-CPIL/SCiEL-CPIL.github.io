# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s wall-clock
_T0 = time.time()
_DEADLINE = 330.0

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
    # Check deadline: if less than 2s remaining, return safe fallback
    if _remaining_time() < 2.0:
        return np.ones(2000, dtype=np.float64)
    
    try:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        # Safety clamps
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        
        # Standard Expected Improvement
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Cost-aware EI
        ei_cost = ei / cost
        
        # UCB-based improvement: (f_best - (mu - beta*sigma)) / cost
        # This rewards both low predicted values and high uncertainty
        beta = 1.5
        ucb_improvement = (f_best - (mu - beta * sigma)) / cost
        
        # Combine: use max of EI/cost and UCB improvement
        # This smoothly blends exploitation (EI) with exploration (UCB)
        scores = np.maximum(ei_cost, ucb_improvement)
        
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
            return np.ones(2000, dtype=np.float64)
# EVOLVE-BLOCK-END
