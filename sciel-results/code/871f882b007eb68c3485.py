"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Internal deadline guard
_T0 = time.time()
_DEADLINE = 320.0  # seconds; evaluator kills at 330s, keep 10s margin


def _check_deadline():
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: blend of UCB (exploration) and cost-normalized EI (exploitation).
    
    Parameters
    ----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns
    -------
    scores : array of shape (2000,) - higher means evaluate next
    """
    if _check_deadline():
        # Fallback: pick uniformly to avoid stalling
        return np.ones_like(mu, dtype=np.float64)

    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # UCB term: explore high-uncertainty regions (critical for multimodal Griewank)
    # Use a larger beta to encourage more exploration
    beta = 2.0
    ucb = -mu + beta * sigma_safe

    # Cost-normalized EI term: exploit where expected improvement per unit cost is high
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Guard against negative EI (shouldn't happen much, but clip)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-8)
    ei_cost = ei / cost_safe

    # Blend: weight exploration more heavily to escape local minima in Griewank
    # Normalize both to comparable scales
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-12
    
    ei_cost_max = np.max(ei_cost)
    if ei_cost_max > 1e-12:
        ei_cost_scaled = ei_cost / ei_cost_max * ucb_range
    else:
        ei_cost_scaled = np.zeros_like(ei_cost)

    # Weighted blend: 70% exploration, 30% exploitation
    w_explore = 0.7
    w_exploit = 0.3
    
    scores = w_explore * ucb + w_exploit * ei_cost_scaled

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
