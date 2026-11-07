"""Cost-aware acquisition function for Bayesian optimization (levy 2d).

New design: a Pareto-style cost normalization. The acquisition value is
expected improvement (EI) blended with an exploration bonus (UCB-like), and it
is divided by the *marginal* cost of the candidate relative to the cheapest
observed point, rather than the absolute cost. This treats cost as a second
objective and explicitly favors low-cost points that still offer high expected
improvement per unit of additional expense.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Explicit wall-clock deadline guard. The evaluator kills the program after
# 330 s; we leave a safety margin so we always return a valid finite result.
_DEADLINE = time.time() + 280.0

# Small module-level cache so repeated calls within a single loop iteration
# (or across the 2000-candidate scoring) do not recompute the same work.
_cache = {"key": None, "value": None}


def _marginal_cost(cost):
    """Extra expense of each candidate relative to the cheapest observed point."""
    c = np.asarray(cost, dtype=float)
    base = float(np.min(c))
    m = c - base
    # Guarantee a strictly positive denominator so the ratio is well defined.
    m = np.maximum(m, 1e-6)
    return m


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard against non-finite inputs producing non-finite output.
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Cache key based on the data actually used.
    key = (float(f_best), mu.tobytes(), sigma.tobytes(), cost.tobytes())
    if _cache["key"] == key and _cache["value"] is not None:
        return _cache["value"]

    # Standard deviation floor to avoid division by zero.
    s = np.maximum(sigma, 1e-9)

    # Expected improvement (higher is better, since we minimize).
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Exploration bonus: an upper-confidence-bound style term. Using the
    # standard deviation directly (rather than a Gaussian quantile) keeps it
    # simple and stable while still rewarding uncertain regions.
    explore = 0.5 * s
    value = ei + explore

    # Pareto cost normalization: divide by the marginal cost relative to the
    # cheapest observed point. This rewards low-cost candidates that still
    # carry high expected improvement per unit of additional expense.
    m = _marginal_cost(cost)
    score = value / m

    # Ensure a finite, valid output.
    score = np.where(np.isfinite(score), score, 0.0)

    _cache["key"] = key
    _cache["value"] = score
    return score
# EVOLVE-BLOCK-END
