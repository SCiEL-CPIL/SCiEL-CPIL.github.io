"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: EI core + Thompson Sampling-inspired exploration.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()

    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best + 1.0)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-10)

    # --- Expected Improvement (core exploitation signal) ---
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.clip(ei, -1e6, 1e6)

    # --- Thompson Sampling-inspired exploration ---
    # Draw a single sample from the posterior for each candidate
    # Use a deterministic seed derived from f_best so that within a run
    # the exploration is consistent, but across different f_best values
    # (i.e., different optimization steps) it varies
    seed_val = int(abs(hash(f_best)) % (2**31))
    rng = np.random.default_rng(seed_val)
    ts_sample = mu + sigma_safe * rng.standard_normal(size=mu.shape)
    # For minimization: lower sampled value is better, so score = -sampled
    ts_scores = -ts_sample

    # --- Adaptive blending ---
    # Measure how "informative" the EI signal is relative to the TS signal
    ei_spread = np.std(ei)
    ts_spread = np.std(ts_scores)

    # If EI has high spread, it's providing strong directional info -> weight it more
    # If EI is flat (all similar), let TS exploration guide
    if ei_spread > 1e-12 and ts_spread > 1e-12:
        # Normalize spreads to get a ratio
        ei_dominance = ei_spread / (ei_spread + ts_spread)
        # Map to weight: when EI dominates, w_ei ~ 0.85; when TS dominates, w_ei ~ 0.55
        w_ei = 0.55 + 0.30 * ei_dominance
    else:
        w_ei = 0.75  # default balanced

    w_ts = 1.0 - w_ei

    # Normalize both components to [0, 1] for fair blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)

    ts_min, ts_max = np.min(ts_scores), np.max(ts_scores)
    ts_range = ts_max - ts_min
    if ts_range > 1e-12:
        ts_norm = (ts_scores - ts_min) / ts_range
    else:
        ts_norm = np.zeros_like(ts_scores)

    # Blend
    scores = w_ei * ei_norm + w_ts * ts_norm

    # --- Penalty for obviously poor regions ---
    # If mu is well above f_best, the candidate is unlikely to improve
    scale = max(abs(f_best), 1.0)
    mu_penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 3.0)
    scores = scores - 0.05 * mu_penalty

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Deadline guard (should be fast; safety net)
    if time.time() - t0 > 325.0:
        pass  # already computed; return immediately

    return scores
# EVOLVE-BLOCK-END
