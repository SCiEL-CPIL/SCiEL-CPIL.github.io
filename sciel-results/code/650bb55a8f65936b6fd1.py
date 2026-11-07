# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_deadline = None

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def _ei_min(mu, sigma, f_best):
    """Expected Improvement for minimization, clipped >= 0, vectorized."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.maximum(ei, 0.0), s

def _pi_min(mu, sigma, f_best):
    """Probability of Improvement for minimization."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    z = (f_best - mu) / s
    pi = norm.cdf(z)
    return np.maximum(pi, 0.0)

def _ucb_score(mu, sigma, beta):
    """LCB-based score (higher = more promising for minimization): -mu + beta*sigma."""
    return -mu + beta * np.maximum(sigma, 1e-12)

def _normalize(a, eps=1e-12):
    lo, hi = np.min(a), np.max(a)
    rng = hi - lo
    if rng < eps:
        return np.ones_like(a) * 0.5
    return (a - lo) / rng

def _greedy_improvement(mu, sigma, f_best, cost, w_ei, w_ucb, w_cost, beta, budget, n_steps=12):
    """
    Estimate the improvement achievable by greedily selecting points using the
    given weighted score under a cost budget. Returns the total improvement
    (f_best_start - final_best), which we want to maximize.
    """
    n = len(mu)
    if n < 2:
        return 0.0

    ei, _ = _ei_min(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, beta)
    cost_eff = 1.0 / (cost + 1e-12)

    ei_n = _normalize(ei)
    ucb_n = _normalize(ucb)
    ce_n = _normalize(cost_eff)

    score = w_ei * ei_n + w_ucb * ucb_n + w_cost * ce_n
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    remaining = budget
    visited = np.zeros(n, dtype=bool)
    best_val = f_best

    for _ in range(min(n_steps, n)):
        feasible = (~visited) & (cost <= remaining)
        if not np.any(feasible):
            break
        cand_score = np.where(feasible, score, -np.inf)
        pick = int(np.argmax(cand_score))
        if not feasible[pick]:
            break
        visited[pick] = True
        remaining -= cost[pick]
        if mu[pick] < best_val:
            best_val = mu[pick]

    return f_best - best_val

def _fit_params(mu, sigma, f_best, cost, budget):
    """
    Fit acquisition parameters (w_ei, w_ucb, w_cost, beta) by maximizing
    greedy improvement over a small grid.
    """
    n = len(mu)
    if n < 2:
        return 0.6, 0.3, 0.1, 1.2

    w_grid = [
        (0.70, 0.25, 0.05),
        (0.60, 0.30, 0.10),
        (0.55, 0.30, 0.15),
        (0.50, 0.35, 0.15),
        (0.45, 0.35, 0.20),
        (0.65, 0.20, 0.15),
        (0.50, 0.40, 0.10),
        (0.75, 0.20, 0.05),
        (0.60, 0.35, 0.05),
        (0.55, 0.35, 0.10),
        (0.40, 0.40, 0.20),
    ]
    beta_grid = [0.8, 1.0, 1.2, 1.5, 1.8, 2.2]

    best_impr = -np.inf
    best_p = (0.6, 0.3, 0.1, 1.2)

    for (we, wu, wc) in w_grid:
        for b in beta_grid:
            impr = _greedy_improvement(mu, sigma, f_best, cost, we, wu, wc, b, budget)
            if impr > best_impr:
                best_impr = impr
                best_p = (we, wu, wc, b)

    return best_p

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that fits a weighted EI + UCB + cost-efficiency
    score against a greedy regret surrogate, then returns continuous, finite,
    higher-is-better scores for all candidates.
    """
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 322.0  # 8s safety margin under 330s kill

    if time.time() > _deadline:
        # Emergency: cheap cost-aware EI, always finite
        ei, _ = _ei_min(mu, sigma, f_best)
        ce = 1.0 / (cost + 1e-12)
        out = 0.8 * ei + 0.2 * ce
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    if n == 0:
        return np.array([])

    # Approximate remaining budget: total budget 30, 5 initial points of cost~1.5 avg => ~7.5 spent
    remaining_budget = 22.5  # nominal

    # Fit parameters against the objective
    try:
        w_ei, w_ucb, w_cost, beta = _fit_params(mu, sigma, f_best, cost, remaining_budget)
    except Exception:
        w_ei, w_ucb, w_cost, beta = 0.6, 0.3, 0.1, 1.2

    # Build final score
    ei, s = _ei_min(mu, sigma, f_best)
    ucb = _ucb_score(mu, sigma, beta)
    cost_eff = 1.0 / (cost + 1e-12)

    ei_n = _normalize(ei)
    ucb_n = _normalize(ucb)
    ce_n = _normalize(cost_eff)

    score = w_ei * ei_n + w_ucb * ucb_n + w_cost * ce_n
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure all values are finite and non-negative
    score = np.maximum(score, 0.0)

    # Scale so max is 1 for stability
    mx = np.max(score)
    if mx > 1e-12:
        score = score / mx

    return score

# EVOLVE-BLOCK-END
