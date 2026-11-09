"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # Safety margin before 300s limit


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and knowledge gradient-like
    exploration, with cost-aware scaling.
    """
    # Check deadline
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_safe = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return fallback_scores

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Replace NaN/Inf with reasonable defaults
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # Compute z-score for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: divide by cost
    cost_ei = ei / cost

    # UCB-like term for exploration
    beta = 0.5
    ucb_term = beta * sigma / cost

    # Combine: cost-aware EI + UCB exploration
    scores = cost_ei + ucb_term

    # Replace any remaining NaN/Inf
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
