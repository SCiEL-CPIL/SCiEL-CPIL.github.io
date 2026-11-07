"""Rewritten acquisition: robust EI + small sigma-scaled exploration bonus, with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global start time for the deadline guard (set on first call).
_START_TIME = None
# Hard wall-clock limit (seconds) for the whole evaluation.
_HARD_LIMIT = 330.0
# Safety margin so we return well before the limit.
_MARGIN = 2.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard: ensure finite inputs; replace non-finite with safe defaults.
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)

    # Standard Expected Improvement (minimization form).
    z = (f_best - mu) / sigma
    # Use clip to keep z in a numerically safe range for cdf/pdf.
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Small exploration bonus: scales with sigma and with the current
    # "optimism gap" so it fades as we converge (sigma -> 0 or mu -> f_best).
    # This nudges the search out of local optima early on.
    # Keep the bonus small so it doesn't dominate exploitation.
    bonus_scale = 0.15
    exploration = bonus_scale * sigma * np.exp(-0.5 * z * z)
    # Fade the exploration bonus as we get close to the best (mu near f_best).
    # When mu <= f_best (improvement found), exploration should shrink.
    fade = np.clip(1.0 - (f_best - mu) / (np.abs(f_best) + 1.0), 0.0, 1.0)
    exploration = exploration * fade

    score = ei + exploration
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard: if we're close to the hard limit, return a safe,
    # deterministic, finite fallback so the loop can finish.
    if _elapsed() > _HARD_LIMIT - _MARGIN:
        # Fall back to pure EI (still finite and valid) without the bonus.
        fallback = np.where(np.isfinite(ei), ei, 0.0)
        return fallback

    return score
# EVOLVE-BLOCK-END
