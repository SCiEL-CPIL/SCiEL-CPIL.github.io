"""Cost-aware acquisition for Hartmann-3D BO loop.
Combines EI/cost (current) with a lightweight calibration over a small
parameter grid (inspiration 1). Calibration selects (gamma, beta, w_ei)
that best ranks low-mu points highly among the top-k scored candidates.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # safety margin before 300s wall-clock limit


def _ei_score(mu, sigma, f_best, cost, beta=1.0):
    """Expected improvement per unit cost^beta."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    cost_safe = np.maximum(cost, 1e-9)
    return ei / (cost_safe ** beta)


def _ucb_score(mu, sigma, cost, gamma=1.0, beta=1.0):
    """UCB-like acquisition: -mu + gamma * sigma, cost-discounted."""
    cost_safe = np.maximum(cost, 1e-9)
    score = (-mu + gamma * sigma) / (cost_safe ** beta)
    return score


def _hybrid_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0, w_ei=0.5):
    """Hybrid of EI and UCB with weighted combination, normalized.
    Uses smooth cost discounting via log-sum-exp for numerical stability.
    """
    w_ucb = 1.0 - w_ei
    ei = _ei_score(mu, sigma, f_best, cost, beta=beta)
    ucb = _ucb_score(mu, sigma, cost, gamma=gamma, beta=beta)

    # Use robust normalization with median and MAD for outlier resistance
    ei_med = np.median(ei)
    ei_mad = np.median(np.abs(ei - ei_med)) + 1e-9
    ei_norm = (ei - ei_med) / ei_mad

    ucb_med = np.median(ucb)
    ucb_mad = np.median(np.abs(ucb - ucb_med)) + 1e-9
    ucb_norm = (ucb - ucb_med) / ucb_mad

    score = w_ei * ei_norm + w_ucb * ucb_norm
    return score


def _calibrate(mu, sigma, f_best, cost, time_budget=2.5):
    """Lightweight calibration over a small grid.
    Proxy: expected improvement per cost, weighted by uncertainty.
    """
    t0 = time.time()

    gamma_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    beta_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
    w_vals = [0.3, 0.5, 0.7]

    best_proxy = -np.inf
    best_params = (1.0, 1.0, 0.5)

    top_k = min(20, max(1, len(mu) // 10))

    # Pre-compute common terms for speed
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_raw = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_raw = np.maximum(ei_raw, 0.0)

    for gamma in gamma_vals:
        for beta in beta_vals:
            for w_ei in w_vals:
                if time.time() - t0 > time_budget:
                    return best_params

                w_ucb = 1.0 - w_ei
                # Fast hybrid score without normalization overhead
                cost_disc = cost_safe ** beta
                ei_disc = ei_raw / cost_disc
                ucb_disc = (-mu + gamma * sigma) / cost_disc

                # Normalize components
                ei_mean = np.mean(ei_disc)
                ei_std = np.std(ei_disc) + 1e-9
                ei_norm = (ei_disc - ei_mean) / ei_std

                ucb_mean = np.mean(ucb_disc)
                ucb_std = np.std(ucb_disc) + 1e-9
                ucb_norm = (ucb_disc - ucb_mean) / ucb_std

                scores = w_ei * ei_norm + w_ucb * ucb_norm

                if not np.all(np.isfinite(scores)):
                    continue

                top_idx = np.argsort(scores)[-top_k:]

                # Better proxy: combine low mu, high sigma (exploration), and low cost
                top_mu = mu[top_idx]
                top_sigma = sigma[top_idx]
                top_cost = cost[top_idx]

                # Proxy: expected improvement potential per cost
                proxy_ei = np.mean(ei_raw[top_idx] / top_cost)
                proxy_exploration = np.mean(top_sigma / top_cost)
                proxy_exploitation = -np.mean(top_mu / top_cost)

                # Weighted combination
                proxy = 0.5 * proxy_ei + 0.3 * proxy_exploration + 0.2 * proxy_exploitation

                if proxy > best_proxy:
                    best_proxy = proxy
                    best_params = (gamma, beta, w_ei)

    return best_params


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Hybrid Acquisition (EI + UCB) with lightweight calibration.

    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs

    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Deadline check
    if time.time() > _DEADLINE:
        # Fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert and sanitize
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)

    mu = np.where(np.isfinite(mu), mu, f_best + 1.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)

    # Calibrate parameters (quick, ~2.5s max)
    # Adaptive time budget: use more time early in the search
    elapsed = time.time() - _START_TIME
    if elapsed < 30:
        cal_time = 3.0
    elif elapsed < 100:
        cal_time = 2.5
    else:
        cal_time = 2.0

    if time.time() < _DEADLINE - 5.0:
        gamma, beta, w_ei = _calibrate(mu, sigma, f_best, cost, time_budget=cal_time)
    else:
        gamma, beta, w_ei = 1.0, 1.0, 0.5

    # Compute hybrid score
    scores = _hybrid_score(mu, sigma, f_best, cost,
                           gamma=gamma, beta=beta, w_ei=w_ei)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
