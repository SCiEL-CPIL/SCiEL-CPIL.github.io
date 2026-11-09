"""Cost-aware acquisition for Levy 2D minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for deadline guarding
_start_time = None
_DEADLINE_SECONDS = 310.0  # 20s safety margin before 330s kill

def _ensure_init():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded():
    return (time.time() - _start_time) > _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    _ensure_init()
    
    # Deadline guard: if we're running out of time, return a simple but safe score
    if _time_exceeded():
        # Return EI-based score without expensive operations
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    n = len(mu)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues: ensure ei is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost-aware: divide by cost
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary strategy: EI per unit cost
    score = ei / cost_safe
    
    # Secondary consideration: boost exploration for very uncertain points
    # This helps escape local minima by favoring high-uncertainty regions
    # when the improvement is small
    uncertainty_bonus = sigma_safe * 0.1
    score = score + uncertainty_bonus / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
