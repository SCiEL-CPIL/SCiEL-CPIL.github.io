"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Combines Expected Improvement (exploitation) with a modest
    uncertainty-driven exploration term, and guards against
    degenerate sigma / extreme z values.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma
    # Clip z to avoid overflow in cdf/pdf
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Exploration bonus: reward uncertainty, scaled to be smaller than
    # typical EI magnitudes.  Use a mild sqrt so very large sigma doesn't
    # dominate.
    exploration = 0.1 * np.sqrt(sigma)

    scores = ei + exploration

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard: if we've spent too long, return a deterministic
    # fallback (pure EI) so the caller still gets valid scores.
    if time.time() > _DEADLINE:
        z2 = (f_best - mu) / sigma
        z2 = np.clip(z2, -10.0, 10.0)
        scores = np.maximum((f_best - mu) * norm.cdf(z2) + sigma * norm.pdf(z2), 0.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
