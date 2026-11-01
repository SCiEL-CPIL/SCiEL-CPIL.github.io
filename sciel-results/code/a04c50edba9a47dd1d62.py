"""Circle packing in perimeter-4 rectangle: discrete cellular decomposition.

Structural change from the seed: instead of optimizing 3N+1 continuous circle
variables (centers + radii + width) with force-relax / SLSQP, we fix a grid
topology and optimize only the (rows+1)*(cols+1) cell-boundary coordinates.
Each circle's radius is exactly the inradius of its rectangular cell, so
disjointness holds by construction and the objective (sum of inradii) is a
smooth, well-conditioned function of the boundary positions.  We search over
several (rows, cols, aspect) configurations and polish each with L-BFGS-B.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _cell_inradii(xs, ys, W, H):
    """Inradius of each cell in a grid defined by boundary arrays xs, ys.

    xs: length cols+1, ys: length rows+1.  Returns (rows, cols) array of
    inradii = min(half-width, half-height) of each cell.
    """
    cw = np.diff(xs)
    ch = np.diff(ys)
    return np.minimum(cw[None, :], ch[:, None]) / 2.0


def _objective(z, rows, cols, W, H):
    """Negative sum of cell inradii for a grid layout.

    z = [x_0..x_cols, y_0..y_rows]  (cols+1 + rows+1 entries).
    x_0=0, x_cols=W are fixed; we optimize the interior boundaries.
    """
    nx = cols + 1
    xs = np.empty(nx)
    xs[0] = 0.0
    xs[-1] = W
    xs[1:-1] = z[:cols - 1]
    ys = np.empty(rows + 1)
    ys[0] = 0.0
    ys[-1] = H
    ys[1:-1] = z[cols - 1:]

    # Keep boundaries in order with a small gap to avoid degenerate cells
    eps = 1e-5
    for i in range(1, nx - 1):
        xs[i] = min(xs[i], W - eps * (nx - 1 - i))
        xs[i] = max(xs[i], eps * i)
    for i in range(1, rows):
        ys[i] = min(ys[i], H - eps * (rows - i))
        ys[i] = max(ys[i], eps * i)

    inr = _cell_inradii(xs, ys, W, H)
    return -float(np.sum(inr))


def _build_from_grid(rows, cols, xs, ys, W, H):
    """Convert a grid layout to the (N,3) circle array, filling row-major."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    inr = _cell_inradii(xs, ys, W, H)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            cx = 0.5 * (xs[j] + xs[j + 1])
            cy = 0.5 * (ys[i] + ys[i + 1])
            rad = inr[i, j]
            x[idx] = cx
            y[idx] = cy
            r[idx] = rad
            idx += 1
    # If the grid has more cells than N, we only use the first N (row-major).
    # If fewer (shouldn't happen since rows*cols>=N), pad with tiny circles.
    if idx < N:
        # place remaining in a corner with tiny radius (shouldn't occur)
        x[idx:] = 1e-4
        y[idx:] = 1e-4
        r[idx:] = 1e-4
    return x, y, r


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _optimize_grid(rows, cols, W, H, deadline, seed=0):
    """Optimize a specific grid layout. Returns (value, x, y, r, W, H) or None."""
    nx = cols + 1
    ny = rows + 1
    n_opt = (cols - 1) + (rows - 1)
    if n_opt <= 0:
        # degenerate single cell: radius = min(W,H)/2, but only 1 circle fits
        return None

    # Initial: uniform grid
    xs0 = np.linspace(0.0, W, nx)
    ys0 = np.linspace(0.0, H, ny)
    z0 = np.concatenate([xs0[1:-1], ys0[1:-1]])

    # Add small jitter for diversity
    rng = np.random.default_rng(seed)
    z0 = z0 + rng.uniform(-0.02, 0.02, size=z0.shape)

    # Bounds: interior x boundaries in (0, W), interior y in (0, H)
    lo = np.concatenate([np.zeros(cols - 1), np.zeros(rows - 1)])
    hi = np.concatenate([np.full(cols - 1, W), np.full(rows - 1, H)])
    bounds = list(zip(lo, hi))

    start = time.time()
    remaining = deadline - start
    if remaining < 2:
        return None

    maxiter = int(min(300, max(50, remaining * 40)))
    try:
        res = minimize(_objective, z0, args=(rows, cols, W, H),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
    except Exception:
        return None

    z = res.x
    xs = np.empty(nx)
    xs[0] = 0.0
    xs[-1] = W
    xs[1:-1] = z[:cols - 1]
    ys = np.empty(ny)
    ys[0] = 0.0
    ys[-1] = H
    ys[1:-1] = z[cols - 1:]

    # Enforce ordering
    eps = 1e-5
    for i in range(1, nx - 1):
        xs[i] = min(xs[i], W - eps * (nx - 1 - i))
        xs[i] = max(xs[i], eps * i)
    for i in range(1, ny - 1):
        ys[i] = min(ys[i], H - eps * (ny - 1 - i))
        ys[i] = max(ys[i], eps * i)

    x, y, r = _build_from_grid(rows, cols, xs, ys, W, H)
    val = float(np.sum(r))

    if _is_valid(x, y, r, W, H):
        return val, x, y, r, W, H
    else:
        # shrink slightly to repair
        r2 = r * 0.99
        if _is_valid(x, y, r2, W, H):
            return float(np.sum(r2)), x, y, r2, W, H
    return None


def _polish(x, y, r, W, H, deadline):
    """Gently grow radii and nudge positions to use remaining space.

    Uses a lightweight force-directed pass plus uniform radius growth.
    """
    n = N
    x = x.copy()
    y = y.copy()
    start = time.time()
    # Force relax to resolve any tiny overlaps and push to walls
    dt = 0.001
    for it in range(300):
        if time.time() > deadline:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - W, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - H, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 15.0
            ux = dx[iu] / d_pairs
            uy = dy[iu] / d_pairs
            for idx in range(len(iu[0])):
                a = iu[0][idx]
                b = iu[1][idx]
                fx[a] += ux[idx] * f_pairs[idx]
                fy[a] += uy[idx] * f_pairs[idx]
                fx[b] -= ux[idx] * f_pairs[idx]
                fy[b] -= uy[idx] * f_pairs[idx]
        x = x + dt * fx
        y = y + dt * fy
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)

    # Grow radii uniformly
    for _ in range(300):
        if time.time() > deadline:
            break
        r_trial = r * 1.0005
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break

    if _is_valid(x, y, r, W, H):
        return x, y, r
    return None


def _grow_individual(x, y, r, W, H, deadline):
    """Grow each circle as much as its local free space allows."""
    n = N
    for i in range(n):
        if time.time() > deadline:
            break
        r_max = min(x[i], W - x[i], y[i], H - y[i])
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            r_max = min(r_max, d - r[j])
        r_max = max(0.0, r_max)
        if r_max > r[i] + 1e-6:
            r[i] = r_max
    if _is_valid(x, y, r, W, H):
        return x, y, r
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    # Candidate grid topologies: (rows, cols). rows*cols must be >= N.
    # We also vary the aspect ratio W (H = 2 - W).
    grid_configs = [
        (4, 6), (5, 5), (3, 7), (6, 4), (5, 4), (7, 3), (4, 5), (6, 4),
        (3, 6), (6, 4), (4, 6), (5, 5), (3, 8), (8, 3), (4, 5), (5, 4),
    ]
    # De-duplicate
    seen = set()
    unique_configs = []
    for cfg in grid_configs:
        if cfg not in seen and cfg[0] * cfg[1] >= N:
            seen.add(cfg)
            unique_configs.append(cfg)

    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.3, 0.7, 1.45, 0.65, 1.6, 0.6, 1.75, 0.55]

    seed_counter = 0
    for W in aspect_ratios:
        if time.time() > deadline - 25:
            break
        H = 2.0 - W
        if H < 0.3:
            continue
        for (rows, cols) in unique_configs:
            if time.time() > deadline - 15:
                break
            # Skip grids whose cell aspect is too extreme for this W/H
            cell_w = W / cols
            cell_h = H / rows
            if min(cell_w, cell_h) < 0.02:
                continue
            seed_counter += 1
            res = _optimize_grid(rows, cols, W, H, deadline, seed=seed_counter)
            if res is not None:
                val, x, y, r, w, h = res
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w, h)

    # Polish the best
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, W, H = best_result
        polished = _polish(x, y, r, W, H, deadline - 5)
        if polished is not None:
            xp, yp, rp = polished
            val = float(np.sum(rp))
            if val > best_value:
                best_value = val
                best_result = (xp, yp, rp, W, H)

    # Individual growth
    if best_result is not None and time.time() < deadline - 3:
        x, y, r, W, H = best_result
        grown = _grow_individual(x, y, r, W, H, deadline - 1)
        if grown is not None:
            xg, yg, rg = grown
            val = float(np.sum(rg))
            if val > best_value:
                best_value = val
                best_result = (xg, yg, rg, W, H)

    if best_result is None:
        # Fallback: uniform 4x6 grid in a 1x1 square
        W, H = 1.0, 1.0
        rows, cols = 4, 6
        xs = np.linspace(0.0, W, cols + 1)
        ys = np.linspace(0.0, H, rows + 1)
        x, y, r = _build_from_grid(rows, cols, xs, ys, W, H)
        return np.column_stack((x, y, r))

    x, y, r, W, H = best_result

    # Final safety shrink to guarantee validity
    for _ in range(20):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
