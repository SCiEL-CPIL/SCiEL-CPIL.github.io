"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE_MARGIN = 10.0  # seconds before the 330s wall-clock kill
_START_TIME = None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: probability of improvement over a threshold,
    normalized by evaluation cost, blended with a smooth LCB term.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Evaluation cost for each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Finite acquisition scores (higher = evaluate next).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard: if we're too close to the kill, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Fallback: simple negative cost-aware EI (fast, always finite)
        safe_sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / safe_sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # --- Probability of Improvement with a threshold (PI-threshold) ---
    # Maximize P(f(x) < f_best - epsilon) = Phi((f_best - epsilon - mu) / sigma)
    # Use a small epsilon to focus on meaningful improvements
    epsilon = 0.05  # fixed threshold offset in objective units
    z_pi = (f_best - epsilon - mu) / sigma
    pi_score = norm.cdf(z_pi)

    # --- Smooth LCB term (differentiable, avoids step-function issues) ---
    # LCB = mu - beta * sigma, where beta balances exploration/exploitation
    # Use a moderate beta; since sigma is in objective units and shekel 4d
    # range is roughly [-10.5, 10], a beta around 1.5-2.5 is reasonable
    beta = 2.0
    lcb_score = mu - beta * sigma

    # Normalize both to [0, 1] range for blending
    # PI score is already in [0, 1]
    # LCB: normalize by the range of observed mu/sigma
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range < 1e-12:
        lcb_norm = np.zeros_like(lcb_score)
    else:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
        # We want lower LCB to be better, so invert: higher score = lower LCB
        lcb_norm = 1.0 - lcb_norm

    # Blend: weight PI-threshold heavily for exploitation, LCB for exploration
    w_pi = 0.7
    w_lcb = 0.3
    blended = w_pi * pi_score + w_lcb * lcb_norm

    # Normalize by cost (higher cost = lower priority)
    scores = blended / cost

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
