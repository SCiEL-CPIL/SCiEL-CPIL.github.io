"""Cost-aware acquisition: UCB-per-cost as the primary signal.
Structural change: drop EI entirely; use UCB (mu + kappa*sigma) divided by
cost. This rewards high-uncertainty regions more aggressively per unit cost,
which the measured cell finding suggests improves simple regret in this
budget-constrained setting.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# --- Explicit deadline guard (330 s hard kill; keep a safety margin) ---
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0   # 10 s safety margin before the 330 s kill

def _elapsed_ok():
    """True while we have at least ~3 s of margin before the hard limit."""
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).

    mu, sigma, f_best, cost : each shape (2000,)
      f_best : scalar, lowest observed value
      cost   : per-candidate cost = 1 + mean(x_scaled)  (in [1, 2])
    Returns 2000 finite scores; higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)

    # Relative sigma floor to keep UCB meaningful when GP is very confident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.05
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # UCB for minimization: lower mu and higher sigma are better.
    # score = (f_best - mu + kappa * sigma) / cost
    # Using f_best - mu instead of -mu keeps scores positive when mu < f_best
    # (which is typical for promising candidates) and avoids negative scores
    # that could confuse the argmax selection.
    kappa = 2.0
    alpha = 1.0

    ucb = (f_best - mu) + kappa * sigma_safe
    cost_pen = cost_safe ** alpha
    score = ucb / cost_pen

    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
