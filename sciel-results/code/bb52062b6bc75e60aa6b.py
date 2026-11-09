"""Rewritten acquisition for branin-2d BO with a deadline guard and a
sampling-based (Thompson-style) exploration mechanism blended with EI."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard (330 s wall-clock limit; keep a safety margin).
_START = time.time()
_DEADLINE = _START + 325.0  # 5 s safety margin

# Branin-2d minimum (given in the task)
_BRANIN_MIN = 0.397887


def _elapsed():
    return time.time() - _START


def _is_dead():
    return _elapsed() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores (higher = evaluate next).

    mu, sigma: shape (2000,), sigma is std-dev in objective units.
    f_best: lowest observed objective value so far.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # Guard against non-finite inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we're over time, just return a deterministic
    # exploitation score (negative mu) to keep the loop moving.
    if _is_dead():
        return -mu.copy()

    # Compute EI (expected improvement) – exploitation signal
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Thompson Sampling: draw one sample from the posterior at each point.
    # sample = mu + sigma * N(0,1).  We want lower objective = better,
    # so the "score" from TS is -sample (higher score = more promising).
    rng = np.random.default_rng()
    eps = rng.standard_normal(n)
    ts_score = -(mu + sigma * eps)

    # Blend EI and TS.  Early in the run (f_best far from optimum) we want
    # more exploration (TS); later we shift to exploitation (EI).
    # Progress metric: how close f_best is to the known minimum.
    # f_best starts around ~10-20 for branin, optimum ~0.398.
    # Map progress in [0,1] where 0 = just started, 1 = very close.
    # Use a robust range: initial f_best is O(10), min is ~0.4.
    f_range = 20.0  # rough scale of initial f_best
    progress = np.clip((f_range - f_best) / (f_range - _BRANIN_MIN), 0.0, 1.0)
    # Blend weight: start with 50% TS / 50% EI, shift to 10% TS / 90% EI
    ts_weight = 0.5 * (1.0 - progress) + 0.1 * progress
    ei_weight = 1.0 - ts_weight

    # Normalize both to comparable scales before blending.
    # EI is non-negative; TS score can be anything.  Shift TS so its mean is 0.
    ts_shifted = ts_score - np.mean(ts_score)
    # Scale EI to have similar magnitude as ts_shifted
    ei_scale = np.std(ts_shifted) + 1e-12
    ei_shifted = (ei - np.mean(ei)) / (np.std(ei) + 1e-12) * ei_scale

    score = ei_weight * ei_shifted + ts_weight * ts_shifted

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
