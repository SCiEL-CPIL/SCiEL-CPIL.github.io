"""Acquisition function for Bayesian optimization of Beale 2D.

Uses a blend of Expected Improvement and a PI-weighted sigma term.
The PI-weighted sigma provides a smoother exploration signal than pure
UCB, giving a better trade-off between exploitation and exploration.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-12)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety: ensure finite, positive sigma.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # EI component.
    ei = _expected_improvement(mu, sigma, f_best)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # PI-weighted sigma term: use the probability of improvement as a
    # weight on sigma, giving a smoother exploration signal.
    z = (f_best - mu) / (sigma + 1e-12)
    pi_weight = norm.cdf(z)
    pi_sigma = pi_weight * sigma

    # Normalize both components to [0, 1] for a stable blend.
    def _normalize(a):
        a = np.asarray(a, dtype=float)
        lo = float(np.min(a))
        hi = float(np.max(a))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    # Blend: 70% EI, 30% PI-weighted sigma (negated since we minimize).
    ei_norm = _normalize(ei)
    ps_norm = _normalize(pi_sigma)
    score = 0.7 * ei_norm + 0.3 * ps_norm

    # Final numerical safety: finite, higher = evaluate next.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
