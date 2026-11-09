"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Lower Confidence Bound (LCB) acquisition function.
    
    Uses a structural change from the previous Thompson Sampling approach.
    The exploration term is scaled by the inverse square root of the cost,
    and the final score is normalized by the cost to reflect resource allocation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline - if we're running out of time, return a simple score
    if time.time() > _DEADLINE - 5.0:
        cost_safe = np.maximum(cost, 1e-6)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are safe
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Adaptive beta based on the current best value.
    # High f_best (early stage) -> Higher beta (more exploration)
    # Low f_best (late stage) -> Lower beta (more exploitation)
    # Rastrigin max is ~100, min is 0.
    # Map f_best from [0, 20] to [0, 1] for the beta adjustment.
    stage = np.clip(f_best / 20.0, 0.0, 1.0)
    beta = 2.0 + 1.0 * stage
    
    # Calculate the Cost-Adjusted LCB
    # Exploration term: sigma / sqrt(cost)
    # This scales down exploration in high-cost regions.
    sqrt_cost = np.sqrt(cost_safe)
    exploration_term = sigma_safe / sqrt_cost
    
    # LCB: mu - beta * exploration_term
    # We want to minimize this, so we negate it for the score.
    lcb = mu - beta * exploration_term
    
    # Cost-normalized score: -LCB / cost
    # This treats the problem as maximizing the expected improvement per unit cost.
    scores = -lcb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
