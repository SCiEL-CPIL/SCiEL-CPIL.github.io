"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before 330s kill
_EPS = 1e-9


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Uses a dynamic balance between exploitation (EI) and exploration (LCB)
    based on the relative scale of uncertainty. Normalizes both terms to
    comparable scales before combining with adaptive weights.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # Compute standard EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute LCB score (higher is better for minimization)
    std_sigma = np.std(sigma)
    mean_sigma = np.mean(sigma)
    beta = 1.0 + 1.0 * (std_sigma / (mean_sigma + _EPS))
    beta = np.clip(beta, 0.5, 3.0)
    
    lcb_score = -mu + beta * sigma

    # Normalize EI to [0, 1]
    ei_max = np.max(ei) + _EPS
    ei_norm = ei / ei_max
    
    # Normalize LCB to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + _EPS
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Adaptive weights based on coefficient of variation of sigma
    cv = std_sigma / (mean_sigma + _EPS)
    # High CV -> early/uncertain -> weight LCB more
    # Low CV -> late/certain -> weight EI more
    weight_lcb = 0.5 + cv
    weight_ei = 1.0
    
    # Final Score
    scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if time.time() > _DEADLINE:
        z2 = (f_best - mu) / sigma
        z2 = np.clip(z2, -10.0, 10.0)
        scores = np.maximum((f_best - mu) * norm.cdf(z2) + sigma * norm.pdf(z2), 0.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
