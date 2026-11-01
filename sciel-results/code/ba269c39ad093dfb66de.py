import numpy as np
import time
from scipy.optimize import minimize

N = 32
_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)

# ---------------------------------------------------------------------------
# Validation / scoring
# ---------------------------------------------------------------------------
def _validity_check(arr, eps=1e-9):
    """Return True if all circles are disjoint (interiors) and inside [0,1]^2."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - eps:
                return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def _hex_seed():
    """Staggered hexagonal layout (6-6-6-4), scaled to the unit square."""
    centers = []
    for row in range(4):
        n_in = 6 if row < 3 else 4
        y = row
        xoff = 0.5 if (row % 2) else 0.0
        for i in range(n_in):
            centers.append((xoff + i, y))
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    r = np.full(N, 0.0)
    for i in range(N):
        r[i] = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])

def _grid_seed():
    """6x6 grid trimmed to N, equal circles."""
    side = 6
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    return np.array(cells[:N])

def _random_seed(rng):
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.02)
    return np.column_stack([x, y, r])

# ---------------------------------------------------------------------------
# Feasibility machinery
# ---------------------------------------------------------------------------
def _shrink_to_valid(arr):
    """Given a candidate, shrink radii minimally to guarantee validity."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    for i in range(N):
        r[i] = max(0.0, min(r[i], x[i], 1 - x[i], y[i], y[i], 1 - y[i]))
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])

def _laplacian(arr, n_iters=1000, dt=0.002, min_r=0.004, cap=0.28):
    """Grow radii toward a Laplacian (inradius) packing while pushing circles
    apart so that the result is a valid, non-overlapping configuration."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    for it in range(n_iters):
        # target radius = min distance to the four walls
        target = np.minimum.reduce([x, 1 - x, y, 1 - y])
        target = np.clip(target, min_r, cap)
        # grow toward target (slow so walls stay respected)
        r = r + 0.05 * (target - r)
        # resolve overlaps by moving centers apart, radii proportional
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                s = r[i] + r[j]
                if d < s:
                    if d < 1e-9:
                        dx = (i - j) * 1e-6
                        dy = 1e-6
                        d = np.hypot(dx, dy)
                    ux, uy = dx / d, dy / d
                    push = (s - d) * 0.5
                    wi = r[i] / max(s, 1e-9)
                    wj = r[j] / max(s, 1e-9)
                    x[i] += ux * push * wi
                    y[i] += uy * push * wi
                    x[j] -= ux * push * wj
                    y[j] -= uy * push * wj
        # clamp to square (keeps walls feasible)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
    # final safety shrink
    return _shrink_to_valid(np.column_stack([x, y, r]))

# ---------------------------------------------------------------------------
# Vectorized objective / gradient (L-BFGS-B)
# ---------------------------------------------------------------------------
def _objective(v):
    x = v[0:N]; y = v[N:2*N]; r = v[2*N:3*N]
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    s = r[:, None] + r[None, :]
    over = np.maximum(0.0, s - d)
    np.fill_diagonal(over, 0.0)
    pen_pair = 0.5 * np.sum(over * over)
    dmin = np.minimum.reduce([x, 1 - x, y, 1 - y])
    overw = np.maximum(0.0, r - dmin)
    pen_wall = np.sum(overw * overw)
    return obj + 100.0 * pen_pair + 100.0 * pen_wall

def _gradient(v):
    x = v[0:N]; y = v[N:2*N]; r = v[2*N:3*N]
    gx = np.zeros(N); gy = np.zeros(N); gr = np.full(N, -1.0)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    s = r[:, None] + r[None, :]
    over = np.maximum(0.0, s - d)
    np.fill_diagonal(over, 0.0)
    # grad of 0.5*sum(over^2) wrt centers: -over * u (u = (dx,dy)/d)
    safe_d = np.where(d > 1e-12, d, 1.0)
    ux = dx / safe_d
    uy = dy / safe_d
    gx += np.sum(-over * ux, axis=1)
    gy += np.sum(-over * uy, axis=1)
    # wrt radii: sum_j over_ij
    gr += np.sum(over, axis=1)
    # wall penalty
    dmin = np.minimum.reduce([x, 1 - x, y, 1 - y])
    overw = np.maximum(0.0, r - dmin)
    g = 2.0 * overw
    gr += g
    # d/dx of (r - dmin)^2 where dmin = min(x, 1-x, y, 1-y)
    gx += -g * (x <= 1 - x) + g * (x > 1 - x) * (x <= y) * (x <= 1 - y) \
          + g * (x > 1 - x) * (x > y) * (x <= 1 - y) * 0  # placeholder (see below)
    # Recompute cleanly:
    gx = np.zeros(N); gy = np.zeros(N)
    for i in range(N):
        if overw[i] <= 0:
            continue
        vals = np.array([x[i], 1 - x[i], y[i], 1 - y[i]])
        k = int(np.argmin(vals))
        if k == 0:
            gx[i] += -g[i]
        elif k == 1:
            gx[i] += g[i]
        elif k == 2:
            gy[i] += -g[i]
        else:
            gy[i] += g[i]
    # recompute pair gradients for gx,gy (overwrite)
    gx += np.sum(-over * ux, axis=1)
    gy += np.sum(-over * uy, axis=1)
    out = np.empty(3 * N)
    out[0:N] = gx
    out[N:2*N] = gy
    out[2*N:3*N] = gr
    return out

# ---------------------------------------------------------------------------
# Polish
# ---------------------------------------------------------------------------
def _polish(arr, max_iter=5000, ftol=1e-14):
    v0 = arr.ravel()
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.0, 0.5))
    try:
        res = minimize(_objective, v0, method='L-BFGS-B', jac=_gradient,
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10})
        v_opt = res.x
        arr_opt = np.column_stack([v_opt[0:N], v_opt[N:2*N], v_opt[2*N:3*N]])
        return _shrink_to_valid(arr_opt)
    except Exception:
        return _shrink_to_valid(arr)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    global _DEADLINE
    _DEADLINE = t0 + 145.0

    best = _shrink_to_valid(_hex_seed())
    best_score = _score(best)
    base_hex = best.copy()
    base_grid = _shrink_to_valid(_grid_seed())
    grid_score = _score(base_grid)
    if grid_score > best_score:
        best_score = grid_score
        best = base_grid.copy()

    rng = np.random.default_rng(12345)
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.0, 1.0))
    for _ in range(N):
        bounds.append((0.0, 0.5))

    restart = 0
    while time.time() < _DEADLINE - 35:  # leave ~35s for polish
        mode = restart % 4
        if mode == 0:
            x0 = base_hex[:, 0].copy(); y0 = base_hex[:, 1].copy()
            x0 += rng.uniform(-0.04, 0.04, N); y0 += rng.uniform(-0.04, 0.04, N)
            x0 = np.clip(x0, 0, 1); y0 = np.clip(y0, 0, 1)
            r0 = np.full(N, 0.04)
        elif mode == 1:
            arr_init = _random_seed(rng)
            x0, y0, r0 = arr_init[:, 0], arr_init[:, 1], arr_init[:, 2]
        elif mode == 2:
            x0 = base_grid[:, 0].copy(); y0 = base_grid[:, 1].copy()
            x0 += rng.uniform(-0.03, 0.03, N); y0 += rng.uniform(-0.03, 0.03, N)
            x0 = np.clip(x0, 0, 1); y0 = np.clip(y0, 0, 1)
            r0 = np.full(N, 0.04)
        else:
            x0 = best[:, 0].copy() + rng.normal(0, 0.015, N)
            y0 = best[:, 1].copy() + rng.normal(0, 0.015, N)
            x0 = np.clip(x0, 0.02, 0.98); y0 = np.clip(y0, 0.02, 0.98)
            r0 = np.clip(best[:, 2].copy() + rng.normal(0, 0.008, N), 0.01, 0.25)

        arr0 = _laplacian(np.column_stack([x0, y0, r0]),
                          n_iters=600, dt=0.002, min_r=0.004, cap=0.28)
        v0 = arr0.ravel()
        try:
            res = minimize(_objective, v0, method='L-BFGS-B', jac=_gradient,
                           bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
            v_opt = res.x
            arr_opt = np.column_stack([v_opt[0:N], v_opt[N:2*N], v_opt[2*N:3*N]])
            arr_valid = _shrink_to_valid(arr_opt)
            sc = _score(arr_valid)
            if sc > best_score:
                best_score = sc
                best = arr_valid.copy()
        except Exception:
            pass
        restart += 1

    # High-precision polish on the best configuration
    if time.time() < _DEADLINE - 5:
        polished = _polish(best, max_iter=5000, ftol=1e-14)
        ps = _score(polished)
        if ps > best_score:
            best_score = ps
            best = polished

    # Final safety
    best = _shrink_to_valid(best)
    if not _validity_check(best):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    return best
