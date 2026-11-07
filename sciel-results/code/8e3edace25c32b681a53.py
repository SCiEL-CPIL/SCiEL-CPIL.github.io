"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Branin 2D domain
_X1_MIN, _X1_MAX = -5.0, 10.0
_X2_MIN, _X2_MAX = 0.0, 15.0
_BRANIN_MIN = 0.397887


def _branin(x1, x2):
    """Evaluate Branin 2D function (vectorized)."""
    pi = np.pi
    return ((x2 - 5.1 * (pi ** 2) / (4.0 * pi ** 2)) ** 2
            + 5.0 * x1 ** 2 - 6.0 * pi * x1 + pi
            + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * np.cos(x1) + 10.0)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.

    Uses a hybrid of Expected Improvement (EI) and a diversity bonus
    derived from the posterior variance, which helps explore Branin's
    multiple local minima.

    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: best (lowest) observed value so far

    Returns: array of shape (2000,) with scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Clamp sigma
    sigma = np.maximum(sigma, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Lower Confidence Bound (LCB) as an alternative exploration term
    # Use a moderate beta to balance exploitation and exploration
    beta = 1.5
    lcb = mu - beta * sigma

    # Combine EI and LCB.
    # Normalize both to [0,1] range over the candidate set for fair combination.
    ei_min, ei_max = ei.min(), ei.max()
    lcb_min, lcb_max = lcb.min(), lcb.max()

    ei_range = ei_max - ei_min
    lcb_range = lcb_max - lcb_min

    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)

    if lcb_range > 1e-12:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb)

    # Weighted combination: 70% EI, 30% LCB
    score = 0.7 * ei_norm + 0.3 * lcb_norm

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
