# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def heuristics(prizes, weights):
    # prizes, weights: shape (5, 100), uniform in [0,1]
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_c, n = prizes.shape
    cap = np.full(n_c, 25.0)

    # --- LP relaxation to get shadow prices (duals) ---
    # max c'x s.t. W'x <= cap, 0 <= x <= 1
    try:
        c = prizes.sum(axis=0)
        A_ub = weights.T          # (5, 100)
        b_ub = cap
        bounds = [(0.0, 1.0)] * n
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs", options={"presolve": True})
        if res is not None and res.status == 0 and res.ineqlin is not None:
            lam = np.abs(np.asarray(res.ineqlin.marginals, dtype=np.float64))
            lam = np.where(np.isfinite(lam), lam, 0.0)
        else:
            lam = None
    except Exception:
        lam = None

    if lam is None:
        # Fallback: inverse of total weight per constraint
        col_sum = weights.sum(axis=0)
        lam = 1.0 / (weights.sum(axis=1) + 1e-9)

    # Reduced cost: prize minus weighted sum of weights by dual prices
    dual_cost = weights.T @ lam          # (100,)
    reduced = prizes.sum(axis=0) - dual_cost

    # Efficiency: prize per total weight
    col_sum = weights.sum(axis=0) + 1e-9
    eff = prizes.sum(axis=0) / col_sum

    # Pareto dominance: count how many items dominate each item
    # (better prize, worse-or-equal weight in every constraint)
    p = prizes.sum(axis=0)
    w = weights
    dominated = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # items j with p[j] >= p[i] and w[:,j] <= w[:,i] in all constraints, j != i
        better_p = p >= p[i]
        better_w = np.all(w <= w[:, i], axis=0)
        mask = better_p & better_w
        mask[i] = False
        dominated[i] = np.sum(mask)

    # Combine: high reduced cost + high efficiency, penalized by dominance
    red = reduced - reduced.min()
    eff_n = eff - eff.min()
    red_max = red.max() if red.max() > 0 else 1.0
    eff_max = eff_n.max() if eff_n.max() > 0 else 1.0
    dom_pen = 1.0 / (1.0 + 2.0 * dominated)

    score = (red / red_max) + (eff_n / eff_max)
    score = score * dom_pen
    score = score + 1e-9

    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    return score
# EVOLVE-BLOCK-END
