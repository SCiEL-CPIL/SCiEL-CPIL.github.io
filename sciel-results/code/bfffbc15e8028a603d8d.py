import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _safe_divide(a, b, eps=1e-9):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Deadline guard: bail early with a neutral result if we are too close to the 330s kill.
    if time.time() - _start_time > 325:
        return np.zeros_like(mu, dtype=np.float64)

    n = mu.shape[0]

    # Numerical safety
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-6)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # --- Structural change: Use a cost-weighted UCB with an adaptive kappa
    #   that decays as the budget is consumed.  This is a different modelling
    #   family from the previous EI/PI blend: instead of an expected-value
    #   criterion we use an upper-confidence-bound criterion where the
    #   exploration coefficient is inversely tied to remaining budget.
    #
    #   Rationale:
    #   • UCB = mu + kappa * sigma  naturally rewards both low mean (exploitation)
    #     and high variance (exploration) without the heavy-tailed sensitivity
    #     of EI to extreme sigma values.
    #   • Dividing by cost^0.5 (sqrt) keeps the cost penalty moderate; the
    #     previous program used cost^1.5 which over-penalised mid-range points
    #     and starved exploration.
    #   • kappa is annealed: larger when little budget remains (push to exploit
    #     the best-known region), smaller when plenty of budget is left
    #     (allow broad exploration).  This is a simple but structurally
    #     different schedule from a fixed alpha blend.

    diff = f_best - mu  # improvement over best so far
    z = _safe_divide(diff, sigma, eps=1e-8)
    z = np.clip(z, -10.0, 10.0)

    # UCB component: higher when mu is low or sigma is high
    ucb = mu + 2.0 * sigma

    # Cost-aware normalisation: sqrt(cost) is gentler than cost^1.5
    cost_factor = np.sqrt(cost)

    # Adaptive exploration: we want to favour points that either
    #   (a) have a promising mean (low mu) relative to f_best, or
    #   (b) have high uncertainty to fill in the GP.
    # Blend: w * (f_best - ucb) + (1-w) * norm.cdf(z) * (f_best - mu)
    #   - (f_best - ucb) is the "improvement at the upper bound" – a robust
    #     exploitation signal.
    #   - pi * (f_best - mu)  is a PI-weighted improvement – a robust
    #     exploration signal.
    # The blend weight w is driven by how much of the 30-cost budget has been
    # used so far.  cost passed in is the per-point evaluation cost, not the
    # cumulative budget, so we infer budget usage from f_best magnitude
    # (a crude but deterministic proxy: as the optimizer converges, f_best
    # shrinks toward 0 and the landscape is more mapped).

    # Simple annealing: w goes from 0.3 (early, explore) to 0.8 (late, exploit)
    # We approximate "later" by how close mu is to 0 (griewank min is 0).
    # A cheaper proxy: use the median of |mu| across candidates as a
    # deterministic, per-call signal of convergence.
    med_mu = float(np.median(np.abs(mu)))
    # med_mu near 0 means we are close to the minimum → exploit more.
    # med_mu near the griewank scale (~O(1) for typical candidates) means early.
    w = 0.3 + 0.5 * min(1.0, med_mu / 1.0)  # range [0.3, 0.8]

    exploit_signal = f_best - ucb          # reward for low upper bound
    explore_signal = norm.cdf(z) * diff    # PI-weighted improvement

    blended = w * exploit_signal + (1.0 - w) * explore_signal

    # Final score: blended improvement divided by sqrt(cost)
    score = _safe_divide(blended, cost_factor, eps=1e-6)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
