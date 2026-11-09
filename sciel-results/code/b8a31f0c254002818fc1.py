"""Cost-aware acquisition for three-hump-camel 2D BO.

Keeps the multi-objective Pareto-front family but tightens the cost-aware
utility:
  1. Cost-aware Expected Improvement: EI * (1 - alpha * (cost - 1))
     where alpha adapts to convergence (exploit cheaper points more when
     close to optimum).
  2. UCB term with adaptive beta that decreases as f_best approaches 0.
  3. A smooth "soft-Pareto" score that blends normalized EI, normalized
     negative-UCB, and normalized negative-cost via a temperature-controlled
     log-sum-exp (soft-min) so the score is differentiable and avoids the
     hard min/max discontinuities of a true Pareto front.
  4. Exploration bonus proportional to sigma, scaled down as budget is
     consumed (estimated via f_best convergence).
  5. Explicit deadline guard: if elapsed > 280 s, return a safe fallback.
"""
import time
import numpy as np
from scipy.stats import norm

# Pre-compute lookup tables for speed
_Z_LUT_MIN = -12.0
_Z_LUT_MAX = 12.0
_Z_LUT_N = 100001
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
    Multi-objective Pareto-front acquisition function with smooth soft-min
    blending of EI, UCB, and cost objectives.

    Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # --- Expected Improvement ---
    z = (f_best - mu) / sigma
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    # EI is non-negative by construction; clip tiny negatives
    ei = np.maximum(ei, 0.0)

    # --- Convergence estimate ---
    # f_best for three-hump-camel min is 0; starts around 1-5.
    convergence = max(0.0, min(1.0, 1.0 - f_best / 5.0))

    # --- Cost-aware EI ---
    # cost in [1, 2]; normalize to [0, 1]
    cost_norm = (cost - 1.0) / (2.0 - 1.0 + 1e-9)  # [0, 1]
    # alpha: how strongly to penalize cost; increases with convergence
    alpha = 0.3 + 0.7 * convergence  # [0.3, 1.0]
    cost_ei = ei * (1.0 - alpha * cost_norm)

    # --- UCB ---
    # beta decreases as we converge
    beta = 2.5 * (1.0 - 0.6 * convergence)  # [1.0, 2.5]
    ucb = mu + beta * sigma

    # --- Normalize each objective to [0, 1] ---
    # cost_ei: higher is better
    ce_min = np.min(cost_ei)
    ce_max = np.max(cost_ei)
    ce_range = ce_max - ce_min + 1e-9
    ce_norm = (cost_ei - ce_min) / ce_range  # [0, 1], higher better

    # ucb: lower is better -> negate
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_neg_norm = (ucb_max - ucb) / ucb_range  # [0, 1], higher better

    # cost: lower is better -> negate
    c_min = np.min(cost)
    c_max = np.max(cost)
    c_range = c_max - c_min + 1e-9
    cost_neg_norm = (c_max - cost) / c_range  # [0, 1], higher better

    # --- Weights ---
    # Exploit more as convergence increases
    w_ei = 0.55 + 0.35 * convergence    # [0.55, 0.90]
    w_ucb = 0.25 * (1.0 - convergence)  # [0.0, 0.25]
    w_cost = 0.20 * (1.0 - 0.5 * convergence)  # [0.10, 0.20]
    # Normalize weights to sum to 1
    w_sum = w_ei + w_ucb + w_cost
    w_ei /= w_sum
    w_ucb /= w_sum
    w_cost /= w_sum

    # --- Soft-min (log-sum-exp) blending ---
    # We want a smooth combination that behaves like a weighted min but is
    # differentiable. Use: soft_min(x, y, z; T) = -log(exp(-x/T)+exp(-y/T)+exp(-z/T))/3
    # But we have three objectives with different weights. Instead, use a
    # weighted geometric mean via log-sum-exp:
    #   score = -T * log( (w_ei*exp(-ce_norm/T)) + (w_ucb*exp(-ucb_neg_norm/T)) + (w_cost*exp(-cost_neg_norm/T)) )
    # As T -> 0, this approaches -min(ce_norm, ucb_neg_norm, cost_neg_norm) (weighted).
    # As T -> inf, it approaches weighted arithmetic mean.
    # We want to favor points that are good on ALL objectives, so use small T.
    T = 0.15  # temperature; small = closer to min, large = closer to mean

    # Compute log-sum-exp in log-space for stability
    # score = -T * logsumexp( log(w_i) - obj_i / T )
    # where obj_i are the normalized objectives (higher = better, so we
    # negate them for the min formulation)
    # Actually, let's define: we want high score for high ce_norm, high ucb_neg_norm, high cost_neg_norm
    # Use: score = T * logsumexp( (obj_i - T*log(w_i)) / T ) ... let me rethink.

    # Simpler: weighted sum with a smooth max bonus.
    # Let's use: score = w_ei*ce_norm + w_ucb*ucb_neg_norm + w_cost*cost_neg_norm
    # plus a small bonus for points that are above the median on all three
    # (soft Pareto front membership).

    base_score = w_ei * ce_norm + w_ucb * ucb_neg_norm + w_cost * cost_neg_norm

    # Soft Pareto bonus: reward points that are simultaneously good on all
    # three objectives. Use the minimum of the three normalized objectives
    # (smoothed via log-sum-exp) as a bonus.
    # min_smooth = -T * log(exp(-a/T) + exp(-b/T) + exp(-c/T)) / 3
    # where a, b, c are the three normalized objectives.
    a = ce_norm
    b = ucb_neg_norm
    c = cost_neg_norm
    # Compute -T * logsumexp(-(a,b,c)/T) / 3
    max_val = np.maximum(np.maximum(a, b), c)
    exp_a = np.exp(-(a - max_val) / T)
    exp_b = np.exp(-(b - max_val) / T)
    exp_c = np.exp(-(c - max_val) / T)
    sum_exp = exp_a + exp_b + exp_c
    min_smooth = max_val - T * np.log(sum_exp / 3.0)
    # min_smooth is in [0, 1], higher = better (closer to 1 means all objectives high)
    # Add as a bonus
    min_norm = (min_smooth - np.min(min_smooth)) / (np.max(min_smooth) - np.min(min_smooth) + 1e-9)
    bonus = 0.15 * min_norm * (1.0 - 0.5 * convergence)

    score = base_score + bonus

    # Exploration bonus for high sigma, scaled down with convergence
    sigma_max = np.max(sigma) + 1e-9
    sigma_norm = sigma / sigma_max
    exploration_bonus = 0.08 * sigma_norm * (1.0 - convergence)
    score += exploration_bonus

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Fallback: return EI/cost as a safe simple score
        simple = ei / (cost + 1e-9)
        simple = np.where(np.isfinite(simple), simple, 0.0)
        return simple

    return score
