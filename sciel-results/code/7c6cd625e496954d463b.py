# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere function.
    Uses normalized Expected Improvement (EI) with adaptive exploration.
    
    The normalization by scale = max(|f_best|, eps) makes the acquisition
    dimensionless, adapting naturally as f_best improves toward 0.
    A small exploration term provides early-stage diversity that decays
    as the search converges.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is positive for stable computation
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (EI)
    # EI(x) = (fb - mu) * Phi((fb - mu)/sigma) + sigma * phi((fb - mu)/sigma)
    z = (fb - mu) / sigma_safe
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Clip small negative values from numerical error
    ei = np.maximum(ei, 0.0)

    # Normalize EI by the scale of f_best to make it dimensionless.
    # This ensures the acquisition behaves consistently whether f_best
    # is 25 (initial) or 0.01 (near optimum).
    scale = max(abs(fb), 1e-8)
    ei_normalized = ei / scale

    # Exploration term: provides early-stage diversity.
    # Uses a smooth decay based on f_best relative to the domain max (25).
    # When f_best is large (early), exploration is stronger.
    # When f_best approaches 0, exploration vanishes.
    # The term is proportional to sigma/scale, so it only rewards
    # uncertainty relative to the current improvement scale.
    fb_frac = fb / (25.0 + fb)  # Maps [0, inf) to [0, 1), smooth decay
    exploration_weight = 0.15 * fb_frac
    exploration_term = exploration_weight * (sigma_safe / scale)

    # Final score: normalized EI plus exploration bonus
    score = ei_normalized + exploration_term

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # If time is running out, return immediately
    if time.time() > _DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
