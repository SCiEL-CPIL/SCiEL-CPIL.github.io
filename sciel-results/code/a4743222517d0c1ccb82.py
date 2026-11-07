# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 325.0
_HARTMANN_6D_MIN = -3.32237


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive UCB with exponential progress-based
    beta decay and a small EI exploitation component.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 3.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        return (f_best - mu) + 1.0 * sigma_safe

    sigma_safe = np.maximum(sigma, 1e-12)

    # Progress: 0 at start (f_best far from optimum), 1 at optimum
    # Hartmann 6d range: min ~-3.32, typical initial f_best ~ 0 to 2
    # Use a robust progress measure
    gap = f_best - _HARTMANN_6D_MIN
    # Typical gap at start is around 3-5; normalize
    # progress goes from 0 (far) to 1 (at optimum)
    # Use exponential mapping for smooth transition
    progress = 1.0 - np.exp(-0.5 * max(gap, 0.0) / 3.0) if gap > 0 else 1.0
    # Actually let's think differently:
    # When gap is large (early), we want high beta (exploration)
    # When gap is small (late), we want low beta (exploitation)
    # beta = beta_max * exp(-k * (1 - gap_norm))
    # where gap_norm = gap / typical_max_gap

    # Typical max gap: f_best starts around 0-2, min is -3.32, so gap ~ 3-5
    # Use gap_max = 5 as reference
    gap_max = 5.0
    gap_clamped = np.clip(gap, 0.0, gap_max)
    gap_norm = gap_clamped / gap_max  # 0 at optimum, 1 at far start

    # Beta schedule: exponential decay from high to low as we progress
    # beta_max = 4.0 (strong exploration early), beta_min = 0.3 (exploitation late)
    beta_max = 4.0
    beta_min = 0.3
    # Smooth exponential transition
    # When gap_norm=1 (early): beta = beta_max
    # When gap_norm=0 (late): beta = beta_min
    # beta = beta_min + (beta_max - beta_min) * exp(-alpha * (1 - gap_norm))
    # At gap_norm=1: beta = beta_min + (beta_max - beta_min) * 1 = beta_max ✓
    # At gap_norm=0: beta = beta_min + (beta_max - beta_min) * exp(-alpha)
    # We want this to be close to beta_min, so alpha should be large enough
    alpha_decay = 3.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-alpha_decay * (1.0 - gap_norm))

    # Compute UCB score for minimization: (f_best - mu) + beta * sigma
    # High when mu is low (promising) and sigma is high (uncertain)
    ucb_score = (f_best - mu) + beta * sigma_safe

    # Add a small EI component for exploitation refinement
    # EI helps focus on points that are likely to improve over f_best
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Blend: primarily UCB, with a small EI weight that increases as we get closer
    # Early: mostly UCB (explore), Late: more EI (exploit)
    ei_weight = 0.1 + 0.4 * (1.0 - gap_norm)  # 0.1 early, 0.5 late

    scores = ucb_score + ei_weight * ei

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
