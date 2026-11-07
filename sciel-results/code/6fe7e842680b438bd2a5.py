"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 25.0


def _norm_cdf_vec(z):
    return 0.5 * np.erfc(-z / np.sqrt(2.0))


def _norm_pdf_vec(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _ei(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-12)
    return (f_best - mu) * _norm_cdf_vec(z) + sigma * _norm_pdf_vec(z)


def _ucb(mu, sigma, beta):
    return -mu + beta * sigma


def _pi(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-12)
    return _norm_cdf_vec(z)


def _kg(mu, sigma, f_best, kappa=0.5):
    z = (f_best - mu) / (sigma + 1e-12)
    return (f_best - mu) * _norm_cdf_vec(z) + kappa * sigma * _norm_pdf_vec(z)


def _minmax(s):
    s_min = np.min(s)
    s_max = np.max(s)
    if s_max - s_min < 1e-15:
        return np.full_like(s, 0.5)
    return (s - s_min) / (s_max - s_min)


def acquisition(mu, sigma, f_best):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Deadline guard: if we're too close to the limit, return a safe fallback
    if time.time() - _START_TIME > _DEADLINE - _MARGIN:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / (sigma_safe + 1e-12)
        fallback = (f_best - mu) * _norm_cdf_vec(z) + sigma_safe * _norm_pdf_vec(z)
        return np.where(np.isfinite(fallback), fallback, 0.0)

    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute acquisition functions
    ei = _ei(mu, sigma_safe, f_best)
    ucb = _ucb(mu, sigma_safe, 1.5)
    pi = _pi(mu, sigma_safe, f_best)
    kg = _kg(mu, sigma_safe, f_best, 0.5)

    # Normalize to [0, 1]
    ei_n = _minmax(ei)
    ucb_n = _minmax(ucb)
    pi_n = _minmax(pi)
    kg_n = _minmax(kg)

    # Adaptive weights: exploit when f_best is small, explore when large
    # f_best for sphere on [-5,5] is in [0, 25]
    exploit = np.clip(1.0 - f_best / 10.0, 0.2, 1.0)
    explore = 1.0 - exploit

    w_ei = 0.45 * exploit + 0.10 * explore
    w_kg = 0.30 * exploit + 0.10 * explore
    w_pi = 0.20 * exploit + 0.05 * explore
    w_ucb = 0.35 * explore + 0.05 * exploit

    combined = w_ei * ei_n + w_kg * kg_n + w_pi * pi_n + w_ucb * ucb_n
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
# EVOLVE-BLOCK-END
