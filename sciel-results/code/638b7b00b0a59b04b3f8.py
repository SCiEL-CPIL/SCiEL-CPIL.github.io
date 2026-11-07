"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a cost-normalized Lower Confidence Bound (LCB):
        score = -(mu - kappa * sigma) / cost
    
    This explicitly trades off exploitation (low mu) against exploration (high sigma)
    while strictly penalizing expensive evaluations. The division by cost means that
    for the same LCB value, cheaper points get proportionally higher scores.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # seconds, safety margin before 330s kill
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Exploration parameter
    kappa = 1.0
    
    # Cost-normalized LCB:
    # score = -(mu - kappa * sigma) / cost
    # This is equivalent to maximizing (kappa * sigma - mu) / cost
    # Cheap, uncertain, low-mean points score highest.
    lcb = mu - kappa * sigma_safe
    scores = -lcb / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
