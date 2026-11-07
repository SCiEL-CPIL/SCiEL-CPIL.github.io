"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
from scipy.stats import norm, t as student_t
import time as _time

_DEADLINE_S = 285.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _student_t_ei(mu, sigma, f_best, nu=4.0):
    """Student-t Expected Improvement for minimization.
    
    Uses Student-t distribution instead of Gaussian for more robust
    exploration-exploitation tradeoff. Lower nu = heavier tails = more exploration.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -30.0, 30.0)
    
    # Student-t CDF and PDF
    t_cdf = student_t.cdf(z, nu)
    t_pdf = student_t.pdf(z, nu)
    
    # For Student-t EI: E[max(f_best - f, 0)] where f ~ t_nu(mu, sigma)
    # = d * P(f < f_best) + sigma * E[(f_best - f)/sigma | f < f_best]
    # The second term involves the t PDF and a correction factor
    ei = d * t_cdf + sigma_safe * t_pdf
    
    # Where sigma is very small, EI is essentially max(d, 0)
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei


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

    # Compute Student-t Expected Improvement (more robust than Gaussian EI)
    ei = _student_t_ei(mu, sigma, f_best, nu=4.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Adaptive cost-awareness: use a smoother exponential penalty
    # Early: high sigma -> lower penalty (allow exploration)
    # Late: low sigma -> higher penalty (conserve budget)
    mean_sigma = np.mean(sigma)
    # Normalize: large sigma -> 1.0 (early), small sigma -> 0.0 (late)
    sigma_norm = mean_sigma / (mean_sigma + 0.05)
    # Exponential penalty: smoother than power law
    # Penalty multiplier: 1.0 when sigma_norm=1, up to ~2.5 when sigma_norm=0
    penalty_mult = 1.0 + 1.5 * (1.0 - sigma_norm)
    penalty_mult = np.clip(penalty_mult, 1.0, 3.0)
    
    cost_safe = np.maximum(cost, 1.0)
    # Use sqrt(cost) for gentler early penalty, scaled by adaptive multiplier
    cost_factor = np.sqrt(cost_safe) * penalty_mult
    
    ei_cost = ei / cost_factor

    # Exploration boost: small term to avoid premature convergence
    explore_boost = 0.015 * sigma / cost_safe
    
    score = ei_cost + explore_boost
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
