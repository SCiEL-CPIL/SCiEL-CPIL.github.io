"""Rewritten cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: a small expected-improvement-style score with a
    UCB exploration bonus, then discounted by evaluation cost.
    """
    t0 = time.time()
    deadline = t0 + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Standardized potential improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)

    # Core exploitation term: scaled EI (bounded, monotone in (f_best - mu))
    ei_scaled = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # Make it strictly positive to keep score meaningful even when no improvement
    ei_scaled = np.maximum(ei_scaled, 0.0)

    # Exploration term: UCB-style sigma boost
    ucb = 0.8 * sigma_safe

    # Combine: EI dominates when improvement is likely, UCB drives exploration
    base = ei_scaled + ucb

    # Cost discount: value per unit cost
    score = base / cost_safe

    # Sanitize
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
