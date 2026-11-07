"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time as _time

_DEADLINE_S = 245.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, 1.0)

    # --- Student-t Inspired EI (Heuristic) ---
    # Use nu=4 for moderate tail heaviness.
    # Inflate sigma to approximate the heavier tails of a t-distribution
    # using the Gaussian EI formula.
    nu = 4.0
    sigma_inflated = sigma_safe * np.sqrt(nu / (nu - 2.0))
    
    z_t = (f_best - mu) / sigma_inflated
    z_t = np.clip(z_t, -10.0, 10.0)
    
    # Gaussian EI formula with inflated sigma
    ei_t = (f_best - mu) * norm.cdf(z_t) + sigma_inflated * norm.pdf(z_t)
    ei_t = np.where(sigma_safe > 0, ei_t, np.maximum(f_best - mu, 0.0))
    ei_t = np.nan_to_num(ei_t, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Cost-aware Scaling ---
    # Use power 1.0 for maximum budget efficiency.
    # Dividing by cost directly rewards cheap points, allowing more evaluations
    # within the fixed budget, which is critical for finding the global minimum
    # in multimodal landscapes.
    ei_t_aware = ei_t / cost_safe

    # --- Confidence-adaptive Exploitation ---
    # When the model is confident (low sigma), switch to pure exploitation
    # to refine the best found point.
    sigma_ref = 0.5
    # Logit for confidence: high when sigma << sigma_ref
    logit = np.log(sigma_safe / sigma_ref)
    conf = 1.0 / (1.0 + np.exp(logit))
    conf = np.clip(conf, 0.0, 1.0)

    # Exploitation score: predicted improvement, cost-aware
    exploit = np.maximum(f_best - mu, 0.0) / cost_safe

    # Blend weights:
    # Base is the robust exploration score (ei_t_aware).
    # Blend in exploitation when confident.
    # Weight 0.2 is a balance: enough to converge, not so much to get stuck.
    exploit_blend = 0.2 * conf
    
    score = (1.0 - exploit_blend) * ei_t_aware + exploit_blend * exploit

    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
