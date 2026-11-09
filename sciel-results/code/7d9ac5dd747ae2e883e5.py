"""Cost-aware acquisition: robust cost-normalized EI with adaptive exploration.
Structural change: use a cleaner cost-aware EI formulation with an adaptive
exploration parameter based on the ratio of GP uncertainty to improvement,
plus a small UCB blend for robustness. Remove the grid-based diversity penalty
which may be too coarse, and instead rely on the natural diversity of 2000
uniform candidates.
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
    n = len(mu)

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)

    # Relative sigma floor to keep EI meaningful when GP is very confident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.02
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # Compute base EI (expected improvement for minimization)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-aware EI: divide by cost
    ei_cost = ei / cost_safe

    # Adaptive exploration: use UCB with an adaptive kappa
    # kappa is based on the ratio of mean sigma to span
    mean_sigma = float(np.mean(sigma_safe))
    kappa = 1.5 * (mean_sigma / max(span, 1e-9))
    kappa = np.clip(kappa, 0.5, 3.0)
    
    ucb = (f_best - mu) + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)
    ucb_cost = ucb / cost_safe
    ucb_cost = np.where(np.isfinite(ucb_cost), ucb_cost, 0.0)

    # Blend EI and UCB: primarily EI, with a UCB boost
    # Use a fixed blend that works well across different stages
    score = 0.7 * ei_cost + 0.3 * ucb_cost
    
    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
