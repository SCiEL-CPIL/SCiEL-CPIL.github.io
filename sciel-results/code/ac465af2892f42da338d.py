"""Circle packing in unit square: pack N circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# ---- global deadline guard (120s safety margin inside 150s limit) ----
_T0 = time.time()
_DEADLINE = _T0 + 118.0

def _time_left():
    return _DEADLINE - time.time()

def _clamp_inside(x, y, r):
    # enforce circle stays inside [0,1]^2
    r = max(r, 1e-9)
    x = min(max(x, r), 1.0 - r)
    y = min(max(y, r), 1.0 - r)
    return x, y, r

def _valid(arr):
    """Return True if circles are disjoint and inside the square."""
    if arr.shape != (N, 3):
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r < 0) or np.any(x < 0) or np.any(y < 0) or np.any(x > 1) or np.any(y > 1):
        return False
    # pairwise distance >= sum of radii (with tiny tolerance)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    s = r[:, None] + r[None, :]
    if np.any(d < s - 1e-8):
        return False
    return True

def _sum_radii(arr):
    return float(np.sum(arr[:, 2]))

def _grid_seed():
    """Grid of equal circles, first N used (baseline)."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _random_seed(rng):
    """Random placement of N non-overlapping circles inside the square."""
    pts = []
    for _ in range(N):
        for _try in range(4000):
            r = rng.uniform(0.01, 0.16)
            x = rng.uniform(r, 1.0 - r)
            y = rng.uniform(r, 1.0 - r)
            ok = True
            for (px, py, pr) in pts:
                if (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2:
                    ok = False
                    break
            if ok:
                pts.append((x, y, r))
                break
    # if we could not place all, fill remaining with tiny circles in free spots
    if len(pts) < N:
        for _ in range(N - len(pts)):
            for _try in range(2000):
                r = 1e-4
                x = rng.uniform(r, 1.0 - r)
                y = rng.uniform(r, 1.0 - r)
                ok = True
                for (px, py, pr) in pts:
                    if (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2:
                        ok = False
                        break
                if ok:
                    pts.append((x, y, r))
                    break
            if len(pts) >= N:
                break
    while len(pts) < N:
        pts.append((0.5, 0.5, 1e-6))
    return np.array(pts[:N])

def _local_refine(arr, rng, budget):
    """Iteratively grow radii then polish with SLSQP. Returns best valid array."""
    best = arr.copy()
    best_val = _sum_radii(arr)
    n = arr.shape[0]

    # build a fixed list of "close pairs" for the SLSQP constraint set (recomputed per call is fine)
    def _pairs(a):
        p = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(a[i, 0] - a[j, 0], a[i, 1] - a[j, 1])
                if d < a[i, 2] + a[j, 2] + 0.15:
                    p.append((i, j))
        return p

    # ---- Stage 1: radius growth pass (positions fixed, tiny tolerance) ----
    def _grow(a):
        x = a[:, 0].copy(); y = a[:, 1].copy(); r = a[:, 2].copy()
        for _it in range(60):
            if _time_left() < 1.0:
                break
            improved = False
            for i in range(n):
                # max radius i can take given others
                rmax = min(1e-9, x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                rmax = max(rmax, 1e-9)
                for j in range(n):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    lim = d - r[j] - 1e-9
                    if lim < rmax:
                        rmax = lim
                if rmax > r[i] + 1e-9:
                    r[i] = rmax
                    improved = True
            if not improved:
                break
        out = np.column_stack([x, y, r])
        return out

    a = arr.copy()
    for _rep in range(3):
        if _time_left() < 1.0:
            break
        a2 = _grow(a)
        if _valid(a2) and _sum_radii(a2) > best_val:
            best = a2.copy(); best_val = _sum_radii(a2)
        a = a2

    # ---- Stage 2: SLSQP joint polish (positions + radii) ----
    def _slsqp(a0, tol, maxiter):
        x0 = a0.ravel().copy()
        def f(v):
            a = v.reshape(n, 3)
            return -np.sum(a[:, 2])
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2], 'jac': lambda v, i=i: _grad_r(v, i)})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2], 'jac': lambda v, i=i: _grad_x(v, i, -1)})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (1.0 - v[3*i]) - v[3*i+2], 'jac': lambda v, i=i: _grad_x(v, i, 1)})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2], 'jac': lambda v, i=i: _grad_y(v, i, -1)})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (1.0 - v[3*i+1]) - v[3*i+2], 'jac': lambda v, i=i: _grad_y(v, i, 1)})
        pairs = _pairs(a0)
        for (i, j) in pairs:
            cons.append({'type': 'ineq',
                         'fun': lambda v, i=i, j=j: v[3*i]-v[3*j],
                         'jac': lambda v, i=i, j=j: _grad_x(v, i, 1) + _grad_x(v, j, -1)})
            cons.append({'type': 'ineq',
                         'fun': lambda v, i=i, j=j: v[3*i+1]-v[3*j+1],
                         'jac': lambda v, i=i, j=j: _grad_y(v, i, 1) + _grad_y(v, j, -1)})
            cons.append({'type': 'ineq',
                         'fun': lambda v, i=i, j=j: v[3*i+2]+v[3*j+2],
                         'jac': lambda v, i=i, j=j: _grad_r(v, i) + _grad_r(v, j)})
        res = minimize(f, x0, method='SLSQP', jac=_full_jac,
                       constraints=cons, tol=tol, maxiter=maxiter,
                       options={'ftol': tol, 'maxiter': maxiter, 'disp': False})
        return res

    def _grad_x(v, i, sgn):
        g = np.zeros_like(v); g[3*i] = sgn; return g
    def _grad_y(v, i, sgn):
        g = np.zeros_like(v); g[3*i+1] = sgn; return g
    def _grad_r(v, i):
        g = np.zeros_like(v); g[3*i+2] = 1.0; return g
    def _full_jac(v):
        # objective gradient: -1 on each radius
        g = np.zeros_like(v)
        for i in range(n):
            g[3*i+2] = -1.0
        return g

    for tol, mi in [(1e-6, 120), (1e-8, 80)]:
        if _time_left() < 2.0:
            break
        try:
            res = _slsqp(a, tol, mi)
            if res.success or res.fun is not None:
                cand = res.x.reshape(n, 3)
                # clamp inside square
                for i in range(n):
                    cand[i, 0], cand[i, 1], cand[i, 2] = _clamp_inside(cand[i, 0], cand[i, 1], cand[i, 2])
                if _valid(cand) and _sum_radii(cand) > best_val:
                    best = cand.copy(); best_val = _sum_radii(cand)
                    a = cand
        except Exception:
            pass

    # ---- Stage 3: final high-precision polish ----
    if _time_left() > 3.0:
        try:
            res = _slsqp(a, 1e-10, 200)
            cand = res.x.reshape(n, 3)
            for i in range(n):
                cand[i, 0], cand[i, 1], cand[i, 2] = _clamp_inside(cand[i, 0], cand[i, 1], cand[i, 2])
            if _valid(cand) and _sum_radii(cand) > best_val:
                best = cand.copy(); best_val = _sum_radii(cand)
        except Exception:
            pass

    return best

def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 118.0

    rng = np.random.default_rng(12345)

    # baseline
    best = _grid_seed().copy()
    best_val = _sum_radii(best)

    # refine baseline first
    if _time_left() > 10.0:
        refined = _local_refine(best, rng, _time_left())
        if _valid(refined) and _sum_radii(refined) > best_val:
            best = refined; best_val = _sum_radii(refined)

    # random restarts
    attempts = 0
    while _time_left() > 12.0:
        seed = _random_seed(rng)
        if not _valid(seed):
            continue
        attempts += 1
        # quick radius-growth pre-pass to get a decent starting point
        a = seed.copy()
        # grow radii greedily
        x = a[:, 0].copy(); y = a[:, 1].copy(); r = a[:, 2].copy()
        for _it in range(40):
            if _time_left() < 12.0:
                break
            improved = False
            for i in range(N):
                rmax = max(1e-9, min(x[i], 1.0 - x[i], y[i], 1.0 - y[i]))
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(x[i] - x[j], y[i] - y[j])
                    lim = d - r[j] - 1e-9
                    if lim < rmax:
                        rmax = lim
                if rmax > r[i] + 1e-9:
                    r[i] = rmax
                    improved = True
            if not improved:
                break
        a = np.column_stack([x, y, r])
        if not _valid(a):
            a = seed
        # local refine
        cand = _local_refine(a, rng, _time_left())
        if _valid(cand) and _sum_radii(cand) > best_val:
            best = cand.copy(); best_val = _sum_radii(cand)

    # final safety: ensure validity, clamp
    for i in range(N):
        best[i, 0], best[i, 1], best[i, 2] = _clamp_inside(best[i, 0], best[i, 1], best[i, 2])
    if not _valid(best):
        # fallback to grid
        best = _grid_seed()
    return best
