"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement + adaptive exploration bonus.
    
    Uses EI as the primary exploitation signal, augmented with an exploration
    term that is strong early (high f_best) and fades as f_best approaches 0.
    This balances global search in early iterations with local refinement later.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: pure exploitation
        score = -mu
        score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
        return score

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Adaptive exploration weight:
    # Sphere 1D on [-5, 5]: max f is 25, min is 0
    # f_norm = 1 when f_best = 25 (early), 0 when f_best = 0 (late)
    f_norm = min(f_best / 25.0, 1.0)

    # Exploration weight: high early, low late
    # Smooth transition from ~0.6 to ~0.1
    explore_weight = 0.1 + 0.5 * f_norm

    # Exploration score: proportional to uncertainty
    explore_score = explore_weight * sigma_safe

    # Combine: EI dominates when f_best is small, exploration helps when large
    score = EI + explore_score

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
