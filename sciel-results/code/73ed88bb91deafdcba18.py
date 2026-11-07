"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0


def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI/cost with a small sigma-based exploration bonus.
    
    This strategy combines the exploitation of Expected Improvement (EI) with
    a gentle exploration bonus based on uncertainty (sigma). The primary signal
    is EI normalized by the evaluation cost, which favors points that offer
    high improvement per unit cost. The exploration bonus is capped to ensure
    it does not dominate the exploitation signal.
    """
    if _check_deadline():
        # Fallback: pick the cheapest point to avoid timeout
        idx = np.argmin(cost)
        scores = np.full_like(mu, -1e10)
        scores[idx] = 1.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to prevent numerical issues in CDF/PDF
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # Primary signal: cost-normalized EI
    # This is the proven high-scoring approach
    ei_cost = ei / cost_safe

    # Small exploration bonus: reward candidates in high-uncertainty regions.
    # We normalize sigma to [0, 1] and scale the bonus by a small fraction of the max EI/cost.
    # This ensures the bonus is a "nudge" rather than a dominant term.
    max_ei_cost = np.max(ei_cost)
    if max_ei_cost > 1e-12:
        sigma_min = np.min(sigma_safe)
        sigma_max = np.max(sigma_safe)
        sigma_range = sigma_max - sigma_min
        if sigma_range > 1e-12:
            sigma_norm = (sigma_safe - sigma_min) / sigma_range
        else:
            sigma_norm = np.full_like(sigma_safe, 0.5)
        
        # Weight: 5% of the max EI/cost value
        # This keeps the exploration term small relative to exploitation
        exploration_bonus = 0.05 * max_ei_cost * sigma_norm
    else:
        exploration_bonus = np.zeros_like(ei_cost)

    scores = ei_cost + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
