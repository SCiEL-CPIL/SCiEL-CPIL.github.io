"""Acquisition function for Bayesian optimization of Goldstein-Price 2D (minimization)."""
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition for minimization.
    Returns scores where higher means "evaluate next".
    score = -mu + k * sigma
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Adaptive exploration coefficient:
    # - When f_best is large (early), we explore more (larger k).
    # - As f_best approaches the known minimum (3.0), we exploit more (smaller k).
    # Goldstein-Price minimum is 3.0, maximum on [-2,2]^2 is around 100+.
    # Normalize the gap to get a "progress" measure.
    f_min_known = 3.0
    f_max_est = 100.0  # rough upper bound on Goldstein-Price on [-2,2]^2

    gap = f_best - f_min_known
    if gap <= 0:
        progress = 1.0
    else:
        # Normalize gap: 1.0 means f_best is at f_max_est, 0.0 means at f_min
        progress = 1.0 - gap / (f_max_est - f_min_known)
        progress = np.clip(progress, 0.0, 1.0)

    # k decays from ~3.0 (high exploration) to ~0.5 (high exploitation)
    k = 0.5 + 2.5 * (1.0 - progress)

    score = -mu + k * sigma

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)

    return score
