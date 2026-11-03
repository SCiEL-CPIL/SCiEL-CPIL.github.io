"""Combined circle packing in unit square.
Uses a structured grid initialization with diverse layouts,
Nelder-Mead optimization with penalty-based objective,
and a deadline guard for safety.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0          # safety margin under 150s kill
START = time.time()


def _elapsed():
    return time.time() - START


def _penalized(x, y, r, w_overlap=20.0, w_bound=50.0):
    """Return a scalar to minimize: -sum(r) + penalties for violations."""
    n = len(r)
    pen = 0.0
    # boundary penalties: circle must fit inside [0,1]^2
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    over = r - d
    if over.size:
        pen += w_bound * np.sum(np.maximum(over, 0.0) ** 2)
    # overlap penalties
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    viol = need - dist
    if viol.size:
        pen += w_overlap * np.sum(np.maximum(viol, 0.0) ** 2)
    return -np.sum(r) + pen


def _valid(x, y, r, eps=1e-6):
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _fix_to_boundary(x, y, r):
    """Clip radii so each circle fits inside the square."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.minimum(r, np.maximum(d, 0.0))


def _grid_init_5x5_plus1(rng):
    """5x5 grid with one extra circle placed in a gap."""
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    # Add one extra near center-top
    extra = [(0.5, 0.1)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.18)
    return x, y, r


def _grid_init_4x6(rng):
    """4x6 grid (24 circles) plus 2 extra."""
    cols, rows = 4, 6
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    # Add 2 extra
    extra = [(0.5, 0.5), (0.25, 0.75)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.19)
    return x, y, r


def _grid_init_6x4(rng):
    """6x4 grid (24 circles) plus 2 extra."""
    cols, rows = 6, 4
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5), (0.75, 0.25)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.19)
    return x, y, r


def _random_init(rng):
    """Random initial positions with small radii."""
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = np.full(N, 0.15)
    return x, y, r


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)

    # Different initial layouts for diversity
    inits = [
        _grid_init_5x5_plus1,
        _grid_init_4x6,
        _grid_init_6x4,
        _random_init,
    ]

    for init_func in inits:
        if _elapsed() > DEADLINE - 5.0:
            break
        x, y, r = init_func(rng)
        r = _fix_to_boundary(x, y, r)

        for restart in range(2):
            if _elapsed() > DEADLINE - 5.0:
                break
            x0 = x.copy()
            y0 = y.copy()
            r0v = r.copy()
            # Add small jitter for multi-start
            jit = rng.normal(0.0, 0.01, size=N)
            x0 = np.clip(x0 + jit, 1e-3, 1 - 1e-3)
            y0 = np.clip(y0 + jit, 1e-3, 1 - 1e-3)
            vec0 = np.concatenate([x0, y0, r0v])

            def objective(v):
                xx = np.clip(v[:N], 1e-6, 1 - 1e-6)
                yy = np.clip(v[N:2*N], 1e-6, 1 - 1e-6)
                rr = np.clip(v[2*N:3*N], 1e-6, 0.5)
                return _penalized(xx, yy, rr)

            try:
                res = minimize(objective, vec0, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                        'maxfev': 4000})
            except Exception:
                continue
            v = res.x
            xx = np.clip(v[:N], 1e-6, 1 - 1e-6)
            yy = np.clip(v[N:2*N], 1e-6, 1 - 1e-6)
            rr = np.clip(v[2*N:3*N], 1e-6, 0.5)
            # Clamp to square to guarantee boundary feasibility
            rr = _fix_to_boundary(xx, yy, rr)
            if _valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

    if best is not None:
        return best

    # Guaranteed fallback: equal grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
