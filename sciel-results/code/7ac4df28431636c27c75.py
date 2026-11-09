"""Cost-aware acquisition for Hartmann-3D BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 290.0  # safety margin before 300s wall-clock limit


def _expected_improvement(mu, sigma, f_best):
    """Standard EI (lower is better)."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Avoid negative EI (numerical)
    ei = np.maximum(ei, 0.0)
    return ei


def _ucb(mu, sigma, f_best, xi=0.0, kappa=2.0):
    """UCB acquisition for minimization: -mu + kappa*sigma."""
    return -mu + kappa * sigma


def _cost_adjusted_ei(mu, sigma, f_best, cost, xi=0.05):
    """
    Cost-aware acquisition: combine EI with a penalty for high cost.
    Higher score = more desirable.
    """
    ei = _expected_improvement(mu, sigma, f_best)
    # Cost penalty: subtract a fraction of cost to prefer cheap points
    # Normalize cost to [0,1] roughly (cost is in [1, 4] since x in [0,1]^3)
    cost_norm = cost / 4.0  # rough normalization
    score = ei - xi * cost_norm
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just EI per cost
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / np.maximum(cost, 1e-9)
    
    # Handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-12)
    
    # Primary strategy: cost-adjusted EI with a small exploration bonus
    # The idea: prefer points with high expected improvement but penalize
    # expensive evaluations. We also add a small UCB-like term to encourage
    # exploration of uncertain regions.
    
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Cost normalization: cost ranges from 1 (all x=0) to 4 (all x=1)
    # Average cost is about 2.5
    cost_penalty = 0.08 * cost  # tune this weight
    
    # Small exploration bonus based on sigma
    exploration = 0.1 * sigma
    
    score = ei - cost_penalty + exploration
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
