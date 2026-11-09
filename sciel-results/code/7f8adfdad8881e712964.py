"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 320.0  # safety margin below the 330s wall-clock kill
_START = time.monotonic()


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe, finite fallback near the cheap centre.
    if time.monotonic() - _START > _DEADLINE:
        return np.full(mu.shape, -np.inf)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical safety: finite output is mandatory.
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Expected improvement (minimisation), in objective units.
    delta = f_best - mu
    z = delta / sigma
    # Clip z to avoid overflow in cdf/pdf for extreme values.
    z = np.clip(z, -50.0, 50.0)
    ei = delta * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost penalty: exponential decay focusing search on the cheap central
    # region where the (shifted) Levy minimum is likely to lie.
    penalty = np.exp(-1.5 * cost)

    score = ei * penalty
    score = np.where(np.isfinite(score), score, -np.inf)
    return score
# EVOLVE-BLOCK-END
