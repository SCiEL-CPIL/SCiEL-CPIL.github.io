"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Uses a portfolio of acquisition functions (EI, UCB, PI) combined
    via normalized averaging to balance exploitation and exploration.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # 1. Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma
    z_ei = np.clip(z_ei, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)

    # 2. Upper Confidence Bound (UCB) for minimization (LCB score)
    # Using a fixed beta=2.0 which is often robust for Hartmann 3D
    beta = 2.0
    lcb = -mu + beta * sigma

    # 3. Probability of Improvement (PI)
    z_pi = (f_best - mu) / sigma
    z_pi = np.clip(z_pi, -10.0, 10.0)
    pi = norm.cdf(z_pi)

    # Normalize each acquisition to [0, 1] to make them comparable
    def normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min + _EPS
        return (x - x_min) / x_range

    ei_norm = normalize(ei)
    lcb_norm = normalize(lcb)
    pi_norm = normalize(pi)

    # Combine using a weighted average
    # Weights: EI is primary, LCB for exploration, PI for confidence
    w_ei = 0.4
    w_lcb = 0.4
    w_pi = 0.2

    scores = w_ei * ei_norm + w_lcb * lcb_norm + w_pi * pi_norm

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to pure EI
        scores = ei
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
