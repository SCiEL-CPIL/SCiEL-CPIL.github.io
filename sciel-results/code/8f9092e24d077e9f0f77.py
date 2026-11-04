"""Circle packing: smooth multi-start L-BFGS-B with final polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_START = time.time()
_DEADLINE = 150.0
_EPS = 1e-9


def _objective(p):
    """Sum radii minus soft penalties (penalize overlap and boundary violation)."""
    n = len(p)
    x = p[:2 * n]
    r = p[2 * n:]
    val = np.sum(r)
    # boundary penalty: circle must fit in [0,1]^2
    b = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    val -= 100.0 * np.sum(b)
    # overlap penalty: for each pair, require dist >= r_i + r_j
    for i in range(n):
        dx = x[i] - x[i + 1:n]
        dy = x[i + n:] - x[i + n + 1:n + (n - i - 1)]
        d = np.sqrt(dx * dx + dy * dy)
        s = r[i] + r[i + 1:n]
        pen = np.maximum(0.0, s - d)
        val -= 100.0 * np.sum(pen)
    return val


def _polish(p):
    """Hard-constraint local optimization: maximize sum r subject to constraints."""
    n = len(p)
    x = p[:2 * n].copy()
    r = p[2 * n:].copy()

    def f(v):
        return -np.sum(v[2 * n:])

    cons = []
    for i in range(n):
        # left
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n + i]})
        # bottom
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i + n] - v[2 * n + i]})
        # right
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2 * n + i]})
        # top
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i + n] - v[2 * n + i]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j:
                    np.sqrt((v[i] - v[j]) ** 2 + (v[i + n] - v[j + n]) ** 2)
                    - v[2 * n + i] - v[2 * n + j],
            })
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    res = minimize(f, p, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-9})
    if res.success or (hasattr(res, 'fun') and res.fun < f(p)):
        return res.x
    return p


def _valid_and_score(arr):
    """Return sum of radii if valid (disjoint, in square), else -inf."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    if np.any(r <= 0) or np.any(np.isnan(arr)):
        return -np.inf
    if np.any(x < -_EPS) or np.any(x > 1 + _EPS) or np.any(y < -_EPS) or np.any(y > 1 + _EPS):
        return -np.inf
    if np.any(r < -_EPS) or np.any(x + r > 1 + _EPS) or np.any(x - r < -_EPS) \
            or np.any(y + r > 1 + _EPS) or np.any(y - r < -_EPS):
        return -np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if d < r[i] + r[j] - 1e-7:
                return -np.inf
    return float(np.sum(r))


def construct():
    n = N
    best_arr = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)

    # ---- Multi-start smooth optimization ----
    starts = 0
    per_start_time = 4.0
    while time.time() < _DEADLINE - 20.0:
        # random init: positions in [0.05, 0.95], radii small
        x = rng.uniform(0.05, 0.95, size=n)
        y = rng.uniform(0.05, 0.95, size=n)
        r = rng.uniform(0.02, 0.15, size=n)
        p0 = np.concatenate([x, y, r])
        res = minimize(_objective, p0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        p = res.x
        arr = np.column_stack([p[:n], p[n:2 * n], p[2 * n:]])
        # clip tiny negatives
        arr[:, 0] = np.clip(arr[:, 0], 0, 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, 1)
        arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
        val = _valid_and_score(arr)
        if val > best_val:
            best_val = val
            best_arr = arr.copy()
        starts += 1
        if starts >= 40:
            break

    # ---- Polish the best candidate with hard constraints ----
    if best_arr is not None:
        if time.time() < _DEADLINE - 5.0:
            p = np.concatenate([best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]])
            pp = _polish(p)
            arr = np.column_stack([pp[:n], pp[n:2 * n], pp[2 * n:]])
            arr[:, 0] = np.clip(arr[:, 0], 0, 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, 1)
            arr[:, 2] = np.clip(arr[:, 2], 1e-4, 0.5)
            val = _valid_and_score(arr)
            if val > best_val:
                best_val = val
                best_arr = arr.copy()

    # ---- Fallback: ensure we return something valid ----
    if best_arr is None or not np.isfinite(best_val):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])

    # final safety clip
    best_arr[:, 0] = np.clip(best_arr[:, 0], 0, 1)
    best_arr[:, 1] = np.clip(best_arr[:, 1], 0, 1)
    best_arr[:, 2] = np.clip(best_arr[:, 2], 1e-4, 0.5)
    return best_arr
