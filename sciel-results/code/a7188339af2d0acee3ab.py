"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 120.0  # wall-clock seconds
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _valid(c):
    """Check circles are inside [0,1]^2 and non-overlapping (with tiny tolerance)."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 3):
        return False
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    if np.any(r <= 0) or np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    if np.any(r > x) or np.any(r > 1 - x) or np.any(r > y) or np.any(r > 1 - y):
        return False
    # pairwise distance >= r_i + r_j
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    req = r[:, None] + r[None, :]
    if np.any(d < req - 1e-7):
        return False
    return True

def _score(c):
    if not _valid(c):
        return 0.0
    return float(np.sum(c[:, 2]))

def _penalty(c, lam=50.0):
    """Smooth penalty for overlaps + boundary. Returns (objective, constraints)."""
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    # objective: minimize -sum(r)
    obj = -np.sum(r)
    cons = []
    # boundary: r <= min(x, 1-x, y, 1-y)
    m = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    cons.append({'type': 'ineq', 'fun': lambda c_, m=m: m - c_[:, 2]})
    # pairwise: dist - (r_i+r_j) >= 0
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    d = np.sqrt(dx * dx + dy * dy)
    rr = r[iu[0]] + r[iu[1]]
    cons.append({'type': 'ineq', 'fun': lambda c_: np.sqrt(
        (c_[:, 0][iu[0]] - c_[:, 0][iu[1]]) ** 2 +
        (c_[:, 1][iu[0]] - c_[:, 1][iu[1]]) ** 2) - (c_[:, 2][iu[0]] + c_[:, 2][iu[1]])})
    return obj, cons

def _force_relax(c, iters=2000, dt=0.002):
    """Force-directed: repulsion when overlapping, attraction to grow. Returns improved c."""
    c = c.copy()
    x, y, r = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
    for it in range(iters):
        if _remaining() < 5.0:
            break
        fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                gap = d - (r[i] + r[j])
                if gap < 0:
                    # strong repulsion when overlapping
                    f = 10.0 * (-gap) ** 1.5 + 0.5
                    fx[i] += f * dx / d; fy[i] += f * dy / d
                    fx[j] -= f * dx / d; fy[j] -= f * dy / d
                    # push radii apart
                    fr[i] -= 0.3 * (-gap); fr[j] -= 0.3 * (-gap)
        # boundary forces
        for i in range(N):
            if x[i] < r[i]: fx[i] += 5.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]: fx[i] -= 5.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]: fy[i] += 5.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]: fy[i] -= 5.0 * (y[i] - (1 - r[i]))
        x += dt * fx
        y += dt * fy
        r += dt * fr
        # clamp
        r = np.maximum(r, 1e-4)
        # keep inside: shrink r to fit boundary
        m = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(m, 1e-4))
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
    c[:, 0] = x; c[:, 1] = y; c[:, 2] = r
    return c

def _slsqp_polish(c, maxiter=300):
    """SLSQP to maximize sum r subject to constraints."""
    x, y, r = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
    def fun(z):
        return -np.sum(z[2 * N:])
    def con_bounds(z):
        # return array of constraint values (>=0)
        xx, yy, rr = z[:N], z[N:2 * N], z[2 * N:]
        vals = []
        m = np.minimum(np.minimum(xx, 1 - xx), np.minimum(yy, 1 - yy))
        vals.append(m - rr)
        iu = np.triu_indices(N, k=1)
        d = np.sqrt((xx[iu[0]] - xx[iu[1]]) ** 2 + (yy[iu[0]] - yy[iu[1]]) ** 2)
        vals.append(d - (rr[iu[0]] + rr[iu[1]]))
        return np.concatenate(vals)
    cons = [{'type': 'ineq', 'fun': con_bounds}]
    bounds = [(0, 1) for _ in range(2 * N)] + [(1e-4, 0.5) for _ in range(N)]
    z0 = np.concatenate([x, y, r])
    res = minimize(fun, z0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False})
    if res.success or res.fun < fun(z0):
        z = res.x
        return np.stack([z[:N], z[N:2 * N], z[2 * N:]], axis=1)
    return c

def _nm_radius_polish(c, maxiter=200):
    """Nelder-Mead on radii only (positions fixed) to squeeze slack."""
    x, y, r = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
    def neg_sum(rr):
        rr = np.maximum(rr, 1e-6)
        # check validity quickly; if invalid, heavy penalty
        m = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        if np.any(rr > m + 1e-9):
            return 1e6 + np.sum(rr)
        iu = np.triu_indices(N, k=1)
        d = np.sqrt((x[iu[0]] - x[iu[1]]) ** 2 + (y[iu[0]] - y[iu[1]]) ** 2)
        if np.any(d < (rr[iu[0]] + rr[iu[1]]) - 1e-9):
            return 1e6 + np.sum(rr)
        return -np.sum(rr)
    res = minimize(neg_sum, r, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-12, 'disp': False})
    rr = np.maximum(res.x, 1e-6)
    # final clamp to boundary
    m = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    rr = np.minimum(rr, m)
    out = c.copy(); out[:, 2] = rr
    return out

def _make_initial(seed):
    rng = np.random.default_rng(seed)
    # random feasible-ish start: small circles scattered
    r0 = 0.02 + 0.03 * rng.random(N)
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    c = np.stack([x, y, r0], axis=1)
    # quick force relax to de-overlap
    c = _force_relax(c, iters=400, dt=0.003)
    return c

def construct():
    best = None
    best_score = -1.0
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for s in seeds:
        if _remaining() < 8.0:
            break
        c = _make_initial(s)
        # phase-based: grow-then-pack
        c = _force_relax(c, iters=1500, dt=0.002)
        sc = _score(c)
        if sc > best_score:
            best = c.copy(); best_score = sc
        if _remaining() < 6.0:
            break
        # SLSQP polish
        c2 = _slsqp_polish(c, maxiter=250)
        sc2 = _score(c2)
        if sc2 > best_score:
            best = c2.copy(); best_score = sc2
        if _remaining() < 4.0:
            break
        # radius NM polish
        c3 = _nm_radius_polish(c2, maxiter=150)
        sc3 = _score(c3)
        if sc3 > best_score:
            best = c3.copy(); best_score = sc3
        if _remaining() < 3.0:
            break
    if best is None or not _valid(best):
        # fallback to a valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    return best

# EVOLVE-BLOCK-END
