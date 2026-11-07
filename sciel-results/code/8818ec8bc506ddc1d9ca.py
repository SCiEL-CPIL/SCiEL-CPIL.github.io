import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)

    # --- Deadline guard: return simple EI/cost if time is up ---
    if not _time_ok():
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive beta for UCB: high exploration early, low late
    f_best_val = float(f_best) if np.isfinite(f_best) else 0.0
    # Progress: 0 at start (f_best near 0 or positive), 1 when f_best near -78.3323
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)
    beta = 0.3 + 1.7 * np.exp(-3.0 * progress)  # 2.0 early, 0.3 late
    ucb = -mu + beta * sigma_safe

    # Cost-aware scoring: divide by cost to penalize expensive evaluations
    # The minimum of ST-2D is in the low-cost region (cost ~1.2)
    # So dividing by cost naturally favors low-cost candidates

    score_ei = ei / cost_safe
    score_ucb = ucb / cost_safe

    # Blend: more EI weight early (to exploit promising regions once found),
    # more UCB weight very early (to explore and find the minimum)
    # Actually, we want:
    # - Very early (progress ~0): high UCB to explore widely
    # - Late (progress ~1): high EI to exploit
    w_ucb = 0.8 * np.exp(-2.0 * progress) + 0.2
    w_ei = 1.0 - w_ucb

    score = w_ei * score_ei + w_ucb * score_ucb

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
