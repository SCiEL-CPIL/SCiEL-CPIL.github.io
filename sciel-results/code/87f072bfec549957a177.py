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
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cost penalty and small exploration bonus.
    
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
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost penalty: prefer cheaper points. cost is in [1, 2] (1 + mean(x)).
    # Use a gentle multiplicative penalty that slightly favors low-cost points.
    # This gives ~0.909 for cost=2 and 1.0 for cost=1.
    cost_penalty = 1.0 / (1.0 + 0.1 * (cost - 1.0))
    
    # Small exploration bonus for high uncertainty, scaled to be minor
    # so that EI dominance is preserved. This helps avoid premature convergence.
    exploration_bonus = 0.05 * (sigma_safe / (sigma_safe + 0.05))
    
    # Combine: cost-penalized EI plus small exploration bonus
    scores = cost_penalty * (ei + exploration_bonus)
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
