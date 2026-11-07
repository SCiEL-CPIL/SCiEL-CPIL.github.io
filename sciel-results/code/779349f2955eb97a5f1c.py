"""Cost-aware acquisition for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline tracking
_DEADLINE_SET = [False]
_DEADLINE = [0.0]

def _ensure_deadline():
    """Set the deadline once; leave 15s safety margin before 330s kill."""
    if not _DEADLINE_SET[0]:
        _DEADLINE_SET[0] = True
        _DEADLINE[0] = time.time() + 315.0  # 315s = 330s - 15s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement (EI) per unit cost,
    with a small exploration bonus to encourage diverse sampling early on.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: float, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    _ensure_deadline()
    
    # Safety: if we're close to deadline, return a simple, fast score
    if time.time() > _DEADLINE[0] - 5.0:
        # Fallback: pure EI/cost, no extra computation
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Standard EI computation
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration bonus: prefer points with higher uncertainty
    # relative to the cost, to avoid getting stuck in exploitation-only
    # The bonus is scaled by the total budget remaining (we don't know it exactly,
    # but we can use a heuristic: larger sigma means more to learn)
    # Use a normalized exploration term
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 1e-9 else 1.0
    explore_bonus = 0.05 * (sigma_safe / sigma_max) / cost_safe
    
    scores = ei_per_cost + explore_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
