"""Rewritten acquisition function with cost-aware UCB and a time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (330s wall-clock; keep ~10s margin)
_DEADLINE = time.time() + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for Bayesian optimization.
    Returns an array of shape (2000,) with higher values preferred.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to a small positive floor to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)

    # Compute a cost-aware UCB score.
    # We use a blend: exploit (low mu) and explore (high sigma), both discounted by cost.
    # The key idea: points that are cheap and have high uncertainty or low predicted value
    # should be preferred.

    # Normalize cost to [0, 1] range for scaling (lower cost = cheaper = better)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)

    # Invert cost so that cheaper points get higher weight
    # Use a soft discount: 1 / (1 + alpha * cost_norm)
    alpha = 1.0
    cost_discount = 1.0 / (1.0 + alpha * cost_norm)

    # UCB component: mu - kappa * sigma  (for minimization, we want low mu and high sigma)
    # But we need to be careful: in minimization, "improvement" is when mu < f_best.
    # A common cost-aware approach:
    #   score = (f_best - mu) * P(mu < f_best) / cost
    # This is expected improvement per unit cost.
    # Alternatively, a UCB variant:
    #   score = -mu / cost + beta * sigma / cost

    # Let's use a hybrid: expected improvement per cost, which is well-understood.
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values
    z = np.clip(z, -10, 10)

    # Standard normal CDF and PDF
    # Use scipy if available, otherwise use an approximation
    try:
        from scipy.stats import norm as _norm
        cdf = _norm.cdf(z)
        pdf = _norm.pdf(z)
    except ImportError:
        # Approximation for erf-based CDF
        cdf = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

    # Expected Improvement (for minimization): EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma * pdf

    # Cost-aware: divide by cost (higher cost -> lower score)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / cost_safe

    # Apply an additional cost discount to further penalize expensive points
    score = score * cost_discount

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
