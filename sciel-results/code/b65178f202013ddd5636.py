"""Cost-aware acquisition for 2D three-hump camel BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.

    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard: if we're running out of time, return something valid quickly
    t0 = time.time()

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware: divide by cost to get improvement per unit cost
    # But we also want to penalize high-cost points more aggressively
    # Use a power law on cost to make the penalty stronger
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected improvement per unit cost
    scores = ei / cost_safe

    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima
    # The bonus is scaled by the inverse cost so it's still cost-aware
    exploration_weight = 0.1
    scores += exploration_weight * sigma_safe / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Quick deadline check
    if time.time() - t0 > 280:
        return scores

    return scores
# EVOLVE-BLOCK-END
