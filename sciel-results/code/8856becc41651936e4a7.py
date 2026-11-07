# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 300.0  # wall-clock limit; keep a safety margin
_START = time.time()


def _elapsed():
    return time.time() - _START


def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are close to the wall, return a safe deterministic score.
    if _elapsed() > _DEADLINE - 2.0:
        # fall back to a cheap, finite score (exploit best mu, penalize cost)
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return (f_best - mu) / cost

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize: ensure finite, positive cost and non-negative sigma.
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)

    # Cost-aware EI (exploitation) blended with cost-aware LCB (exploration).
    # EI per unit cost.
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei_cost = ei / cost

    # LCB per unit cost: prefer low mu and high uncertainty, discounted by cost.
    # kappa tuned for a 2D offset optimum; moderate exploration.
    kappa = 1.6
    lcb = (f_best - mu - kappa * sigma) / cost
    lcb = np.where(np.isfinite(lcb), lcb, 0.0)

    # Blend: EI is the primary signal; LCB adds exploration in cheap, uncertain areas.
    score = 0.65 * ei_cost + 0.35 * lcb

    # Ensure finiteness and a finite spread.
    score = np.where(np.isfinite(score), score, 0.0)

    # A tiny floor so that near-zero-EI cheap points still get a chance early,
    # but does not dominate the signal.
    score = score + 1e-3 / cost

    return score
# EVOLVE-BLOCK-END
