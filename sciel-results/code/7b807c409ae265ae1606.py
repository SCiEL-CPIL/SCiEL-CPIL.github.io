import numpy as np
from scipy.stats import norm
import time

# Deadline guard: kill after 320s to stay safely under 330s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _check_deadline():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with
    sublinear cost penalty (1/sqrt(cost)) and an uncertainty bonus.
    
    Uses Thompson Sampling-like structure with EI as the primary exploitation
    signal, combined with a probability-of-improvement exploration term.
    The sublinear cost penalty allows exploration of high-cost regions that
    may contain the global minimum.
    """
    if not _check_deadline():
        # Fallback: return simple regret-based scores
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        scores = (f_best - mu) / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Sublinear cost penalty: 1/sqrt(cost) allows exploration of high-cost regions
    cost_penalty = 1.0 / np.sqrt(cost_safe)

    # Expected Improvement (minimization: improvement = f_best - f(x))
    # EI = sigma * (z * Phi(z) + phi(z)) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    # EI is already in objective units, non-negative
    ei = np.maximum(ei, 0.0)

    # Component 2: Probability of improvement weighted by uncertainty / cost
    # This adds exploration pressure in uncertain regions that have a
    # reasonable chance of containing a new minimum.
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 0.0, 1.0)
    exploration = p_improve * sigma_safe

    # Combine: EI dominates when confident, exploration fills in when uncertain
    # Both scaled by sublinear cost penalty
    w_ei = 1.0
    w_exp = 0.4
    scores = (w_ei * ei + w_exp * exploration) * cost_penalty

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
