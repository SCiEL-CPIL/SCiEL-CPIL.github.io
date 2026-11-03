"""Circle packing in unit square: maximize sum of radii of N=26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

# ---------------------------------------------------------------------------
# Smooth geometry helpers
# ---------------------------------------------------------------------------

def _wall_terms(c, r):
    """Smooth (log-sum-exp) of the 4 wall clearances for a circle."""
    x, y = c
    clear = np.array([x - r, y - r, (1.0 - x) - r, (1.0 - y) - r])
    m = clear.max()
    return m + np.log(np.sum(np.exp(clear - m)))


def _pair_term(ci, ri, cj, rj):
    """Smooth (log-sum-exp) of (distance - ri - rj) for a pair."""
    d = np.linalg.norm(ci - cj)
    return d - (ri + rj)


def _objective_and_penalty(c, r, beta):
    """
    Returns (score, penalty, nviol).
    score = sum(r) - (1/beta)*soft_penalty  (we maximize this)
    soft_penalty = sum over circles of logsumexp(wall_terms)
                   + sum over pairs of (pair_term)   [pair term already smooth]
    """
    n = len(r)
    soft_pen = 0.0
    for i in range(n):
        soft_pen += _wall_terms(c[i], r[i])
    for i in range(n):
        for j in range(i + 1, n):
            soft_pen += _pair_term(c[i], r[i], c[j], r[j])
    score = np.sum(r) - (1.0 / beta) * soft_pen
    # hard violation count for bookkeeping
    nviol = 0
    for i in range(n):
        x, y = c[i]
        if x - r[i] < -1e-9 or y - r[i] < -1e-9 or (1.0 - x) - r[i] < -1e-9 or (1.0 - y) - r[i] < -1e-9:
            nviol += 1
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(c[i] - c[j]) - r[i] - r[j] < -1e-9:
                nviol += 1
    return score, soft_pen, nviol


def _is_feasible(c, r, tol=1e-7):
    for i in range(len(r)):
        x, y = c[i]
        if x - r[i] < -tol or y - r[i] < -tol or (1.0 - x) - r[i] < -tol or (1.0 - y) - r[i] < -tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if np.linalg.norm(c[i] - c[j]) - r[i] - r[j] < -tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _grid_init():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    base_r = 0.5 / side - 1e-3
    c = np.array(cells[:N], dtype=float)
    r = np.full(N, base_r, dtype=float)
    return c, r


def _random_init(rng):
    # random centers in [0.15, 0.85], small-ish radii
    c = rng.uniform(0.15, 0.85, size=(N, 2))
    r = rng.uniform(0.03, 0.07, size=N)
    return c, r


def _grid_jitter_init(rng, jitter=0.05):
    c, r = _grid_init()
    c = c + rng.uniform(-jitter, jitter, size=(N, 2))
    c = np.clip(c, 0.05, 0.95)
    r = r * rng.uniform(0.8, 1.2, size=N)
    return c, r


# ---------------------------------------------------------------------------
# Optimizer (maximize score via L-BFGS-B on negated objective)
# ---------------------------------------------------------------------------

def _optimize(c0, r0, beta, maxiter, deadline, polish=False):
    """Run L-BFGS-B to maximize the smooth score. Returns (c, r, best_score, feasible)."""
    n = len(r0)

    def neg_score(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        score, _, _ = _objective_and_penalty(c, r, beta)
        return -score

    best_c, best_r = c0.copy(), r0.copy()
    best_score = _objective_and_penalty(c0, r0, beta)[0]

    def cb(xk):
        nonlocal best_c, best_r, best_score
        c = xk[:2 * n].reshape(n, 2)
        r = xk[2 * n:]
        sc, _, _ = _objective_and_penalty(c, r, beta)
        if sc > best_score:
            best_score = sc
            best_c = c.copy()
            best_r = r.copy()
        return time.time() > deadline

    x0 = np.concatenate([c0.ravel(), r0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    res = minimize(neg_score, x0, method="L-BFGS-B",
                   bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-12 if polish else 1e-10,
                            "gtol": 1e-8 if polish else 1e-6, "maxls": 40},
                   callback=cb)
    c = res.x[:2 * n].reshape(n, 2)
    r = res.x[2 * n:]
    sc, _, _ = _objective_and_penalty(c, r, beta)
    if sc > best_score:
        best_score, best_c, best_r = sc, c.copy(), r.copy()
    return best_c, best_r, best_score, _is_feasible(best_c, best_r)


# ---------------------------------------------------------------------------
# Repair: minimize maximum hard violation to force feasibility
# ---------------------------------------------------------------------------

def _repair(c0, r0, deadline):
    n = len(r0)

    def max_viol(z):
        c = z[:2 * n].reshape(n, 2)
        r = z[2 * n:]
        v = 0.0
        for i in range(n):
            x, y = c[i]
            v = max(v, r[i] - x, r[i] - y, r[i] - (1.0 - x), r[i] - (1.0 - y))
        for i in range(n):
            for j in range(i + 1, n):
                v = max(v, r[i] + r[j] - np.linalg.norm(c[i] - c[j]))
        return v

    x0 = np.concatenate([c0.ravel(), r0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    res = minimize(max_viol, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 400, "ftol": 1e-14, "gtol": 1e-8,
                            "maxls": 40, "maxfun": 20000},
                   callback=lambda xk: time.time() > deadline)
    c = res.x[:2 * n].reshape(n, 2)
    r = res.x[2 * n:]
    return c, r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def construct():
    rng = np.random.default_rng(12345)
    deadline = _DEADLINE

    best_c, best_r = None, None
    best_sum = -1.0

    def _consider(c, r):
        nonlocal best_c, best_r, best_sum
        if _is_feasible(c, r):
            s = float(np.sum(r))
            if s > best_sum:
                best_sum = s
                best_c, best_r = c.copy(), r.copy()

    # candidate initializations
    inits = [_grid_init()]
    for _ in range(6):
        inits.append(_grid_jitter_init(rng, jitter=rng.uniform(0.02, 0.08)))
    for _ in range(3):
        inits.append(_random_init(rng))

    # Phase 1: smooth objective optimization with several betas (anneal)
    betas = [20.0, 40.0, 80.0, 150.0]
    for c0, r0 in inits:
        if time.time() > deadline - 15:
            break
        cur_c, cur_r = c0.copy(), r0.copy()
        for beta in betas:
            if time.time() > deadline - 15:
                break
            c1, r1, _, _ = _optimize(cur_c, cur_r, beta, maxiter=300, deadline=deadline)
            cur_c, cur_r = c1, r1
        _consider(cur_c, cur_r)

    # If we have a feasible best, polish it with high precision
    if best_c is not None and time.time() < deadline - 20:
        for beta in [150.0, 300.0]:
            if time.time() > deadline - 15:
                break
            c1, r1, _, _ = _optimize(best_c, best_r, beta, maxiter=500, deadline=deadline, polish=True)
            _consider(c1, r1)

    # Guarantee feasibility: if no feasible found, or to be safe, run repair on best
    if best_c is None:
        # fall back to repairing the grid
        c1, r1 = _repair(*_grid_init(), deadline)
        best_c, best_r = c1, r1
    else:
        # try to repair the current best to ensure strict feasibility (cheap)
        if time.time() < deadline - 10:
            c1, r1 = _repair(best_c, best_r, deadline)
            if _is_feasible(c1, r1):
                if float(np.sum(r1)) >= best_sum - 1e-6:
                    best_c, best_r = c1, r1
            # else keep previous feasible best

    # Final safety: ensure strict feasibility with tiny shrink if needed
    if not _is_feasible(best_c, best_r, tol=1e-7):
        # shrink radii slightly and re-repair
        c1, r1 = _repair(best_c, best_r * 0.98, deadline)
        best_c, best_r = c1, r1
    # clamp tiny numerical negatives
    best_r = np.clip(best_r, 1e-9, None)
    best_c = np.clip(best_c, 0.0, 1.0)

    return np.column_stack([best_c, best_r])
