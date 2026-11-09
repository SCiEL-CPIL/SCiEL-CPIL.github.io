"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 330.0  # seconds, hard kill
_START_TIME = None

def _ensure_deadline():
    """Track elapsed time and raise if we're approaching the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 30:  # 30s safety margin
        raise TimeoutError("Approaching deadline")

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (POI)
    with an exploration bonus.
    
    Parameters:
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviation predictions
    f_best: float - Best observed value so far
    cost: array of shape (N,) - Cost of evaluating each point
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    _ensure_deadline()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Exploration bonus: reward uncertainty, scaled by cost
    # This encourages exploring regions with high uncertainty, but penalized by cost
    # Use a small weight to balance exploitation (POI) and exploration (sigma)
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * sigma_safe / np.maximum(cost, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: POI divided by cost plus exploration bonus
    scores = poi / cost_safe + exploration_bonus
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
