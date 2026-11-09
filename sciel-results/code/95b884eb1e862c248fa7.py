# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_TOTAL_BUDGET = 330.0
_SAFETY_MARGIN = 5.0
_CALL_COUNT = 0


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _deadline_check(safety_margin=_SAFETY_MARGIN):
    """Return True if we are safe to continue."""
    return _get_elapsed() < _TOTAL_BUDGET - safety_margin


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    Blends Thompson Sampling (randomized posterior predictive) with
    sigma-inflated Expected Improvement, using a time-adaptive schedule
    that shifts from exploration-heavy to exploitation-heavy.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    t = _CALL_COUNT

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)

    # --- Thompson Sampling component ---
    # Draw a single sample from the posterior predictive: mu - sigma * z
    # (we want to minimize, so lower sampled values are better)
    # Negate to convert to a "higher is better" score
    z_sample = np.random.standard_normal(len(mu))
    ts_score = -(mu - sigma * z_sample)

    # --- Sigma-inflated Expected Improvement component ---
    # Moderate inflation to keep EI landscape informative
    inflation = 1.5
    sigma_eff = sigma * inflation
    sigma_safe = np.where(sigma_eff > 1e-9, sigma_eff, 1e-9)

    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Normalize both components to [0, 1] for meaningful blending ---
    ts_min, ts_max = np.min(ts_score), np.max(ts_score)
    ei_min, ei_max = np.min(ei), np.max(ei)

    ts_range = ts_max - ts_min if ts_max > ts_min else 1.0
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0

    ts_norm = (ts_score - ts_min) / ts_range
    ei_norm = (ei - ei_min) / ei_range

    # --- Adaptive weighting ---
    # Early: more Thompson Sampling (broad randomized exploration)
    # Late: more EI (targeted exploitation near the minimum)
    # The transition is smooth and controlled by the call count
    w_ts = np.exp(-t / 8.0)
    w_ei = 1.0 - w_ts

    # Blend the two normalized scores
    score = w_ei * ei_norm + w_ts * ts_norm

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
