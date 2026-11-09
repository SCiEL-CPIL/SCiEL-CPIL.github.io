"""Acquisition for Bayesian optimization of weierstrass 1d (Pareto distance-to-ideal)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Explicit deadline guard: track elapsed time across calls within a run.
_T0 = time.time()
_DEADLINE_MARGIN_S = 10.0   # safety margin before the 300s/330s kill
_DEADLINE = _T0 + 330.0 - _DEADLINE_MARGIN_S


def _within_deadline():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best):
    """Pareto distance-to-ideal-point acquisition.

    mu, sigma: shape (2000,) arrays in objective units (sigma is a std dev).
    Returns 2000 finite scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    f_best = float(f_best)

    # Numerical safety: ensure finite, non-negative sigma.
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Objective 1: we want low f, so the "good" direction is low mu.
    # Work in a normalized, bounded space so the ideal point (0,0) is reachable.
    # g1: predicted improvement magnitude (positive when mu < f_best), clipped >= 0.
    # g2: uncertainty, clipped >= 0.
    g1 = np.clip(f_best - mu, 0.0, None)
    g2 = np.clip(sigma, 0.0, None)

    # Normalize each objective by its current max so the ideal point (0,0)
    # is the corner of a unit box; distance to that corner encodes the trade-off.
    m1 = float(np.max(g1)) if g1.size else 0.0
    m2 = float(np.max(g2)) if g2.size else 0.0
    if not np.isfinite(m1) or m1 <= 0.0:
        m1 = 1.0
    if not np.isfinite(m2) or m2 <= 0.0:
        m2 = 1.0

    n1 = g1 / m1
    n2 = g2 / m2

    # Distance to the ideal point (0,0) in normalized objective space.
    # A candidate that is both low-mean and high-uncertainty sits near the origin.
    d = np.sqrt(n1 * n1 + n2 * n2)

    # Higher score = closer to ideal. Invert distance and shift so scores are positive.
    scores = 1.0 / (d + 1e-9)

    # Ensure strictly finite output (defensive).
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
