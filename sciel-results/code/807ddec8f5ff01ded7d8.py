"""Circle packing in unit square: maximize sum of radii for N disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_TIME_BUDGET = _DEADLINE - 20.0  # safety margin


def _pairwise(i, j, p, r):
    dx = p[2*i] - p[2*j]
    dy = p[2*i+1] - p[2*j+1]
    d = np.hypot(dx, dy)
    return d - (r[i] + r[j])


def _soft_violation(p, r, beta):
    """Smooth measure of constraint violation (>=0 when feasible-ish)."""
    n = len(r)
    # containment: r_i <= min(x,1-x,y,1-y)  -> violation = r_i - margin
    x = p[0::2]; y = p[1::2]
    margin = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    cont_v = r - margin
    # pair overlap: r_i + r_j - dist  (positive = overlap)
    pair_v = np.zeros(n*(n-1)//2)
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = p[2*i] - p[2*j]
            dy = p[2*i+1] - p[2*j+1]
            d = np.hypot(dx, dy)
            pair_v[k] = r[i] + r[j] - d
            k += 1
    v = np.concatenate([cont_v, pair_v])
    # soft-max: logsumexp(beta * v) / beta, clipped
    v = np.maximum(v, 0.0)
    if np.max(v) == 0.0:
        return 0.0
    m = np.max(v)
    return (m + np.log(np.sum(np.exp(beta*(v - m)))))/beta


def _objective(p, beta, weight):
    n = N
    r = p[2*n:]
    r = np.maximum(r, 1e-6)
    sumr = np.sum(r)
    viol = _soft_violation(p, r, beta)
    return -(sumr) + weight*viol


def _is_valid(p, tol=1e-6):
    n = N
    r = p[2*n:]
    if np.any(r <= 0):
        return False
    x = p[0::2]; y = p[1::2]
    if np.any(r > np.minimum(np.minimum(x,1-x),np.minimum(y,1-y)) + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _score(p):
    n = N
    r = p[2*n:]
    return float(np.sum(r))


def _init_random(seed):
    rng = np.random.default_rng(seed)
    p = np.zeros(2*N + N)
    p[0::2] = rng.uniform(0.05, 0.95, N)
    p[1::2] = rng.uniform(0.05, 0.95, N)
    p[2*N:] = rng.uniform(0.03, 0.08, N)
    return p


def _init_grid(seed):
    p = np.zeros(2*N + N)
    side = int(np.ceil(np.sqrt(N)))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            p[2*idx] = (i + 0.5)/side
            p[2*idx+1] = (j + 0.5)/side
            p[2*N+idx] = 0.5/side - 1e-3
            idx += 1
    return p


def construct():
    t0 = time.time()
    best_p = _init_grid(0)
    best_score = _score(best_p)

    inits = [_init_grid(0)]
    for s in range(1, 40):
        inits.append(_init_random(s*97 + 13))

    # anneal beta / weight
    beta_schedule = [20.0, 40.0, 80.0, 160.0, 320.0]
    weight_schedule = [1.0, 2.0, 4.0, 8.0, 16.0]

    for bi, (beta, weight) in enumerate(zip(beta_schedule, weight_schedule)):
        if time.time() - t0 > _TIME_BUDGET:
            break
        for init in inits:
            if time.time() - t0 > _TIME_BUDGET:
                break
            try:
                res = minimize(
                    _objective, init.copy(),
                    args=(beta, weight),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)]*(2*N) + [(1e-6, 0.5)]*N,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                p = res.x
                if _is_valid(p):
                    sc = _score(p)
                    if sc > best_score:
                        best_score = sc
                        best_p = p.copy()
            except Exception:
                pass

    # final high-precision polish with very high weight
    if time.time() - t0 < _TIME_BUDGET:
        for beta, weight in [(640.0, 32.0), (1280.0, 64.0)]:
            if time.time() - t0 > _TIME_BUDGET:
                break
            try:
                res = minimize(
                    _objective, best_p.copy(),
                    args=(beta, weight),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)]*(2*N) + [(1e-6, 0.5)]*N,
                    options={'maxiter': 400, 'ftol': 1e-14}
                )
                p = res.x
                if _is_valid(p):
                    sc = _score(p)
                    if sc > best_score:
                        best_score = sc
                        best_p = p.copy()
            except Exception:
                pass

    n = N
    r = best_p[2*n:]
    x = best_p[0::2]
    y = best_p[1::2]
    out = np.column_stack([x, y, r])
    return out
