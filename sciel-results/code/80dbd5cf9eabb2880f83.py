"""Circle packing in perimeter-4 rectangle: SLSQP + vectorized forces + fine grid + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0

def _compute_distances(x, y):
    """Vectorized pairwise distance matrix."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _objective_fast(z):
    """Fast objective with vectorized overlap/boundary penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3:
        return 1e6

    obj = -np.sum(r)
    pw = 500.0

    # Vectorized pairwise overlap penalty
    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    overlap = np.maximum(0, r_sum - d)
    # Only count upper triangle (avoid double counting)
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    obj += pw * np.sum(overlap[triu_mask] ** 2)

    # Boundary penalties
    bl_x = np.maximum(0, r - x)
    br_x = np.maximum(0, x - (w - r))
    bl_y = np.maximum(0, r - y)
    br_y = np.maximum(0, y - (h - r))
    obj += pw * (np.sum(bl_x ** 2) + np.sum(br_x ** 2) +
                 np.sum(bl_y ** 2) + np.sum(br_y ** 2))

    # Aspect ratio penalty
    if w < 0.5:
        obj += pw * (0.5 - w) ** 2
    elif w > 1.5:
        obj += pw * (w - 1.5) ** 2

    return obj

def _check_validity(z, tol=1e-9):
    """Check if packing is valid, return (valid, value)."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w <= 0.05 or h <= 0.05:
        return False, 0.0

    for i in range(N):
        if r[i] <= 0:
            return False, 0.0
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False, 0.0
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False, 0.0

    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[triu_mask] < r_sum[triu_mask] - tol):
        return False, 0.0

    val = float(np.sum(r))
    return True, val

def _force_relax(z, max_iter=300, dt=0.003):
    """Force-directed relaxation to resolve overlaps."""
    z = z.copy().astype(float)
    n = N

    for _ in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        x = z[0::3][:n].copy()
        y = z[1::3][:n].copy()
        r = z[2::3][:n].copy()
        w = z[-1]
        h = 2.0 - w

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 20
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 20
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 20
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 20

        # Pairwise repulsion (vectorized)
        d = _compute_distances(x, y)
        r_sum = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < r_sum[i, j] and d[i, j] > 1e-12:
                    overlap = r_sum[i, j] - d[i, j]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    fx[i] += (dx / d[i, j]) * overlap * 10
                    fy[i] += (dy / d[i, j]) * overlap * 10
                    fx[j] -= (dx / d[i, j]) * overlap * 10
                    fy[j] -= (dy / d[i, j]) * overlap * 10

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z

def _polish_result(z, time_budget):
    """Iterative radius growth polish with position adjustments."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    start = time.time()
    max_iter = 2000
    dt_pos = 0.0005

    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        # Position adjustment for overlaps
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 30
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 30
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 30
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 30

        d = _compute_distances(x, y)
        r_sum = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < r_sum[i, j] and d[i, j] > 1e-12:
                    overlap = r_sum[i, j] - d[i, j]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    fx[i] += (dx / d[i, j]) * overlap * 15
                    fy[i] += (dy / d[i, j]) * overlap * 15
                    fx[j] -= (dx / d[i, j]) * overlap * 15
                    fy[j] -= (dy / d[i, j]) * overlap * 15

        x += dt_pos * fx
        y += dt_pos * fy
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

        # Try to grow each circle
        d = _compute_distances(x, y)
        for i in range(n):
            max_growth = min(x[i] - r[i], w - x[i] - r[i],
                            y[i] - r[i], h - y[i] - r[i])
            for j in range(n):
                if i != j:
                    max_growth = min(max_growth, d[i, j] - r[i] - r[j])
            if max_growth > 1e-8:
                r[i] += min(0.002, max_growth * 0.4)

    z[0::3][:n] = x
    z[1::3][:n] = y
    z[2::3][:n] = r
    return z

def _fine_grid_start(w0, h0, grid_res=60):
    """Create initial guess from fine grid sampling."""
    # Place circles on a fine grid, pick best N positions
    positions = []
    step_x = w0 / grid_res
    step_y = h0 / grid_res
    r0 = min(step_x, step_y) * 0.48

    for i in range(grid_res):
        for j in range(grid_res):
            cx = (i + 0.5) * step_x
            cy = (j + 0.5) * step_y
            positions.append((cx, cy))

    # Greedy selection: pick positions that maximize min distance
    selected = []
    remaining = list(range(len(positions)))
    rng = np.random.default_rng(123)
    rng.shuffle(remaining)

    # Start with a few random picks spread out
    for _ in range(min(5, N)):
        idx = remaining.pop(0)
        selected.append(positions[idx])

    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        # Sample a subset for speed
        sample = remaining[:min(200, len(remaining))]
        for idx in sample:
            px, py = positions[idx]
            min_d = min(np.hypot(px - sx, py - sy) for sx, sy in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        if best_idx >= 0:
            remaining.remove(best_idx)
            selected.append(positions[best_idx])
        else:
            break

    z = []
    for cx, cy in selected[:N]:
        z.extend([cx, cy, r0])
    z.append(w0)
    return np.array(z)

def construct():
    best_value = 0.0
    best_result = None

    # Generate diverse initial guesses
    guesses = []

    # Fine grid starts with varying aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.65, 1.4, 0.6]
    for w0 in aspect_ratios:
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
        z = _fine_grid_start(w0, h0, grid_res=60)
        guesses.append(z)

    # Standard grid starts
    side = int(np.ceil(np.sqrt(N)))
    for w0 in [1.0, 0.85, 1.18, 0.75, 1.33, 0.67]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        z = []
        for i in range(N):
            row, col = divmod(i, side)
            cx = (col + 0.5) / side * w0
            cy = (row + 0.5) / side * h0
            r0 = min(w0 / side, h0 / side) * 0.45
            z.extend([cx, cy, r0])
        z.append(w0)
        guesses.append(np.array(z))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(6):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = []
        for i in range(N):
            cx = rng.uniform(0.05, w0 - 0.05)
            cy = rng.uniform(0.05, h0 - 0.05)
            r0 = rng.uniform(0.04, 0.10)
            z.extend([cx, cy, r0])
        z.append(w0)
        guesses.append(np.array(z))

    # Shuffle to interleave types
    rng.shuffle(guesses)

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    for i, z0 in enumerate(guesses):
        elapsed = time.time() - START
        if elapsed > DEADLINE - 8:
            break

        remaining = DEADLINE - elapsed - 5
        if remaining < 2:
            break

        # Force-directed relaxation
        z0_relaxed = _force_relax(z0, max_iter=200)

        # SLSQP optimization
        maxiter = int(min(200, remaining * 25))
        if maxiter < 10:
            maxiter = 10
        try:
            res = minimize(_objective_fast, z0_relaxed, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 30})

            z_opt = res.x
            valid, val = _check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Repair: shrink radii
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.93
                valid, val = _check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # Phase 2: Polish best result
    if best_result is not None:
        remaining = DEADLINE - (time.time() - START) - 5
        if remaining > 3:
            polished = _polish_result(best_result, remaining)
            valid, val = _check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

    # Phase 3: Final tight polish
    if best_result is not None:
        remaining = DEADLINE - (time.time() - START) - 3
        if remaining > 2:
            # Another SLSQP pass with tighter tolerance
            try:
                res = minimize(_objective_fast, best_result, method='SLSQP',
                               bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-14, 'maxls': 20})
                z_opt = res.x
                valid, val = _check_validity(z_opt)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_opt.copy()
            except Exception:
                pass

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]

    # Safety shrink
    r = r * (1 - 1e-7)

    return np.column_stack((x, y, r))
