import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 285


def _elapsed_ok():
    return (time.time() - _START_TIME) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    n = mu.size

    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Expected improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-normalized EI: divide by cost to get "improvement per unit cost"
    # This is the simplest cost-aware metric
    ei_per_cost = ei / cost

    # Also compute a UCB-like score for exploration
    # For minimization: -mu + kappa * sigma
    # kappa adapts based on how much we've explored
    # Use a moderate exploration constant
    kappa = 2.0
    ucb = -mu + kappa * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)

    # Normalize both scores to [0, 1] for blending
    def _normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        rng = x_max - x_min
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - x_min) / rng

    ei_norm = _normalize(ei_per_cost)
    ucb_norm = _normalize(ucb)

    # Blend: weight EI-per-cost heavily when there's clear improvement signal,
    # fall back to UCB exploration when EI is flat
    ei_max = np.max(ei_per_cost)
    if ei_max > 1e-8:
        # Strong improvement signal available: lean on EI/cost
        w_ei = 0.75
    else:
        # No clear improvement: explore more
        w_ei = 0.35

    w_ucb = 1.0 - w_ei

    score = w_ei * ei_norm + w_ucb * ucb_norm

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    # Fallback for near-timeout
    if not _elapsed_ok():
        fallback = ei / np.power(cost, 0.5)
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    return score
