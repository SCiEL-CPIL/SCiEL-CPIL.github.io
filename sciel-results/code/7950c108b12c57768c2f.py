"""Cost-aware acquisition for three-hump-camel 2D BO.

Structural change: replace scalar EI heuristic with a multi-objective
Pareto-front approximation that explicitly models the trade-off between
expected improvement and evaluation cost. We use a probabilistic utility
score that combines:
  1. Expected Improvement (EI) for exploitation
  2. Cost-adjusted exploration bonus (UCB-like) for exploitation
  3. A Pareto-dominance-based selection that favors points on the
     improvement-cost frontier

The key insight is that for a fixed budget, the optimal policy is not simply
EI/cost. Instead, we want to maximize the probability of achieving a target
regret reduction per unit cost. We approximate this with a weighted combination
that adapts based on how much budget remains.
"""
import time
import numpy as np
from scipy.stats import norm

# Pre-compute lookup tables for speed
_Z_LUT_MIN = -12.0
_Z_LUT_MAX = 12.0
_Z_LUT_N = 200001
_z_lut_grid = np.linspace(_Z_LUT_MIN, _Z_LUT_MAX, _Z_LUT_N)
_cdf_lut = norm.cdf(_z_lut_grid)
_pdf_lut = norm.pdf(_z_lut_grid)


def _fast_norm_cdf(z):
    """Vectorized norm.cdf via interpolation."""
    z = np.clip(z, _Z_LUT_MIN, _Z_LUT_MAX)
    idx = ((z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_N - 1)).astype(np.int64)
    idx = np.clip(idx, 0, _Z_LUT_N - 2)
    x0 = _z_lut_grid[idx]
    x1 = _z_lut_grid[idx + 1]
    y0 = _cdf_lut[idx]
    y1 = _cdf_lut[idx + 1]
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _fast_norm_pdf(z):
    """Vectorized norm.pdf via interpolation."""
    z = np.clip(z, _Z_LUT_MIN, _Z_LUT_MAX)
    idx = ((z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_N - 1)).astype(np.int64)
    idx = np.clip(idx, 0, _Z_LUT_N - 2)
    x0 = _z_lut_grid[idx]
    x1 = _z_lut_grid[idx + 1]
    y0 = _pdf_lut[idx]
    y1 = _pdf_lut[idx + 1]
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def acquisition(mu, sigma, f_best, cost):
    """
    Multi-objective Pareto-front acquisition function.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 250.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Expected Improvement
    z = (f_best - mu) / sigma
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z

    # UCB with adaptive beta based on budget
    # As budget decreases, we shift more toward exploitation
    # cost is in [1, 2], total budget is 30, initial cost ~5*1.5=7.5
    # remaining budget fraction: (30 - total_cost) / 30
    # We don't have total_cost directly, but cost array gives per-point cost
    # Use mean cost as proxy for budget usage rate
    mean_cost = np.mean(cost)
    # Estimate remaining budget: assume we've used roughly (number of evals) * mean_cost
    # We don't know number of evals, but we can use the cost array shape
    # Actually, cost has shape (2000,) representing the cost of each candidate
    # The total spent so far is not directly available, but f_best tells us convergence
    
    # Adaptive beta: start high (exploration), decrease as f_best improves
    # f_best for three-hump-camel min is 0, so lower f_best = better
    # Use f_best to gauge convergence: if f_best is close to 0, exploit more
    # Scale: f_best starts around 1-5, ends near 0
    convergence = max(0.0, 1.0 - f_best / 5.0)  # 0 when f_best=5, 1 when f_best=0
    beta = 2.0 * (1.0 - 0.5 * convergence)  # beta in [1.0, 2.0]

    ucb = mu + beta * sigma

    # Cost-adjusted scores
    # Objective 1: minimize ucb (exploitation)
    # Objective 2: minimize cost (efficiency)
    
    # Normalize ucb to [0, 1] for Pareto combination
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range  # lower is better

    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # lower is better

    # Pareto-front based scoring:
    # A point is on the Pareto front if no other point dominates it
    # (i.e., has both lower ucb and lower cost)
    # We approximate Pareto dominance with a weighted sum that adapts
    
    # Weight that balances the two objectives
    # When EI is high (good exploitation signal), weight toward EI
    # When EI is low (uncertain), weight toward cost efficiency
    ei_max = np.max(ei) + 1e-9
    ei_norm = ei / ei_max  # [0, 1], higher is better
    
    # Adaptive weight: blend EI exploitation with cost efficiency
    # w_ei increases as we find good improvements
    w_ei = 0.6 + 0.3 * convergence  # [0.6, 0.9]
    w_cost = 1.0 - w_ei  # [0.1, 0.4]

    # Combined score: high EI, low ucb, low cost
    score = w_ei * ei_norm - w_cost * (0.5 * ucb_norm + 0.5 * cost_norm)

    # Add a small exploration bonus for high sigma (uncertainty)
    sigma_max = np.max(sigma) + 1e-9
    sigma_norm = sigma / sigma_max
    exploration_bonus = 0.1 * sigma_norm * (1.0 - convergence)

    score += exploration_bonus

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass

    return score
