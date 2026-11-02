"""Circle packing in unit square: SLSQP with explicit constraints + multi-start."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_DEADLINE = 145.0  # wall-clock safety margin (limit is 150s)


def _validate(circles):
    """Check circles are inside unit square with disjoint interiors."""
    if circles is None:
        return False
    c = np.asarray(circles, dtype=float)
    if c.ndim != 2 or c.shape[0] != N or c.shape[1] != 3:
        return False
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _score(circles):
    if not _validate(circles):
        return -np.inf
    return float(np.sum(circles[:, 2]))


def _get_delaunay_edges(circles):
    """Get edges from Delaunay triangulation of centers."""
    c = np.asarray(circles, dtype=float)
    pts = c[:, :2]
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    edges.add((int(simplex[a]), int(simplex[b])))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _slsqp_objective(v, edges):
    """Objective: negative sum of radii (minimize)."""
    r = v[2 * N:]
    return -np.sum(r)


def _slsqp_constraints(edges):
    """Build constraint dict for SLSQP: non-overlap + boundary."""
    cons = []
    # Non-overlap constraints: d_ij >= r_i + r_j
    for (i, j) in edges:
        def overlap_cons(v, i=i, j=j):
            x = v[:N]
            y = v[N:2 * N]
            r = v[2 * N:]
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            return d - r[i] - r[j]
        cons.append({'type': 'ineq', 'fun': overlap_cons})
    # Boundary: x_i - r_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[i] - v[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N + i] - v[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[N + i] - v[2 * N + i]})
    return cons


def _initial_config_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pts = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)][:N]
    r0 = 0.03
    return np.array([[x, y, r0] for (x, y) in pts])


def _initial_config_hex():
    """Hexagonal lattice initialization."""
    pts = []
    s = 0.17
    for j in range(12):
        row_y = (j + 0.5) * s
        offset = (s / 2.0) * (j % 2)
        for i in range(12):
            x = (i + 0.5) * s + offset
            y = row_y
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    if len(pts) < N:
        side = int(np.ceil(np.sqrt(N)))
        pts = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    pts = pts[:N]
    r0 = 0.03
    return np.array([[x, y, r0] for (x, y) in pts])


def _initial_config_rings():
    """Ring-based initialization: center + concentric rings."""
    pts = [(0.5, 0.5)]
    # Ring 1: 6 circles
    for k in range(6):
        angle = k * np.pi / 3
        pts.append((0.5 + 0.18 * np.cos(angle), 0.5 + 0.18 * np.sin(angle)))
    # Ring 2: 12 circles
    for k in range(12):
        angle = k * np.pi / 6 + np.pi / 6
        pts.append((0.5 + 0.35 * np.cos(angle), 0.5 + 0.35 * np.sin(angle)))
    pts = pts[:N]
    r0 = 0.03
    return np.array([[x, y, r0] for (x, y) in pts])


def _initial_config_random(seed=42):
    """Random initialization with jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            y = (j + 0.5) / side + rng.uniform(-0.1, 0.1) / side
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append((x, y))
    pts = pts[:N]
    r0 = 0.03
    return np.array([[x, y, r0] for (x, y) in pts])


def _fix_config(circles):
    """Fix minor constraint violations by clamping positions and reducing radii."""
    c = np.array(circles, dtype=float)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    # Clamp positions
    for i in range(N):
        r[i] = max(r[i], 1e-4)
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    # Resolve overlaps iteratively
    for _pass in range(50):
        fixed = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    fixed = False
                    if d < 1e-12:
                        dx, dy = 1e-5, 0.0
                        d = 1e-5
                    corr = (min_d - d) / (2.0 * d)
                    x[i] += dx * corr
                    y[i] += dy * corr
                    x[j] -= dx * corr
                    y[j] -= dy * corr
                    # Re-clamp
                    x[i] = np.clip(x[i], r[i], 1 - r[i])
                    y[i] = np.clip(y[i], r[i], 1 - r[i])
                    x[j] = np.clip(x[j], r[j], 1 - r[j])
                    y[j] = np.clip(y[j], r[j], 1 - r[j])
        if fixed:
            break
    # If still overlapping, shrink radii
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j]:
                factor = d / (r[i] + r[j]) * 0.99
                r[i] *= factor
                r[j] *= factor
    c = np.column_stack([x, y, r])
    return c


def _slsqp_optimize(circles, max_iter=300):
    """Optimize using SLSQP with explicit constraints. Returns (circles, score)."""
    c = np.array(circles, dtype=float)
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    v0 = np.concatenate([x, y, r])

    # Get edges for constraints
    edges = _get_delaunay_edges(c)
    cons = _slsqp_constraints(edges)

    # Bounds: positions in [0,1], radii in [1e-4, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(1e-4, 0.5)] * N

    try:
        res = minimize(
            _slsqp_objective, v0, args=(edges,),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        v = res.x
        c_new = np.column_stack([v[:N], v[N:2*N], v[2*N:]])
        c_new = _fix_config(c_new)
        s = _score(c_new)
        if np.isfinite(s):
            return c_new, s
    except Exception:
        pass

    # Fallback: return original
    return c, _score(c)


def _penalty_optimize(circles, edges, max_iter=500, lr0=0.01):
    """Penalty-based local optimization as backup."""
    c = np.array(circles, dtype=float)
    x = c[:, 0].copy()
    y = c[:, 1].copy()
    r = c[:, 2].copy()
    n = N
    best = c.copy()
    best_s = _score(c)
    w_pen = 100.0
    lr = lr0

    for it in range(max_iter):
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        gr[:] = 1.0
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            gap = r[i] + r[j] - d
            if gap > 0:
                if d < 1e-12:
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / d, dy / d
                gx[i] -= w_pen * ux
                gy[i] -= w_pen * uy
                gx[j] += w_pen * ux
                gy[j] += w_pen * uy
                gr[i] -= w_pen * 0.5
                gr[j] -= w_pen * 0.5
        for i in range(n):
            if x[i] - r[i] < 0:
                gx[i] += w_pen
                gr[i] -= w_pen
            if x[i] + r[i] > 1:
                gx[i] -= w_pen
                gr[i] -= w_pen
            if y[i] - r[i] < 0:
                gy[i] += w_pen
                gr[i] -= w_pen
            if y[i] + r[i] > 1:
                gy[i] -= w_pen
                gr[i] -= w_pen

        step = lr
        improved = False
        for _ls in range(5):
            nx = x + step * gx
            ny = y + step * gy
            nr = r + step * gr
            nr = np.clip(nr, 1e-4, 0.5)
            cand = np.column_stack([nx, ny, nr])
            s = _score(cand)
            if np.isfinite(s) and s > best_s + 1e-12:
                x, y, r = nx, ny, nr
                best = cand.copy()
                best_s = s
                improved = True
                break
            step *= 0.5
        if not improved:
            lr *= 0.8
            if lr < 1e-6:
                break
        else:
            lr = min(lr0, lr * 1.001)
    return best, best_s


def construct():
    t0 = time.time()
    best = None
    best_s = -np.inf

    # Multiple initializations
    inits = [
        _initial_config_grid(),
        _initial_config_hex(),
        _initial_config_rings(),
        _initial_config_random(42),
        _initial_config_random(123),
    ]

    for init_c in inits:
        if time.time() > _DEADLINE - 15:
            break

        # SLSQP optimization
        c, s = _slsqp_optimize(init_c, max_iter=200)
        if np.isfinite(s) and s > best_s:
            best_s = s
            best = c.copy()

        # If time remains, do penalty polish
        if time.time() < _DEADLINE - 10 and best is not None:
            edges = _get_delaunay_edges(best)
            polished, s2 = _penalty_optimize(best, edges, max_iter=300, lr0=0.005)
            if np.isfinite(s2) and s2 > best_s:
                best_s = s2
                best = polished.copy()

    # Final SLSQP polish on best
    if best is not None and time.time() < _DEADLINE - 5:
        c, s = _slsqp_optimize(best, max_iter=100)
        if np.isfinite(s) and s > best_s:
            best_s = s
            best = c.copy()

    # Ensure validity
    if best is None or not _validate(best):
        best = _fix_config(_initial_config_grid())

    return best
