import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # safety margin before 150s


def _penalty_objective(v, lam_b, lam_o):
    """v: [x(26), y(26), r(26)]. Minimize -sum(r) + penalties."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -r.sum()

    # Boundary penalty
    b = np.maximum(0, r - x) + np.maximum(0, r - (1.0 - x)) + \
        np.maximum(0, r - y) + np.maximum(0, r - (1.0 - y))
    obj += lam_b * b.sum()

    # Overlap penalty (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rsum = r[:, None] + r[None, :]
    # Only upper triangle
    overlap = np.maximum(0, rsum - np.sqrt(np.maximum(0, dist2)))
    # Zero out diagonal
    np.fill_diagonal(overlap, 0.0)
    obj += lam_o * (overlap**2).sum()

    return obj


def _is_valid(arr, tol=1e-7):
    """Check if arrangement is valid (all constraints satisfied)."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _fix_feasibility(arr, max_iter=200):
    """Shrink radii until all constraints are satisfied."""
    x, y, r = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    
    for _ in range(max_iter):
        max_violation = 0.0
        # Boundary violations
        for i in range(N):
            d = min(x[i], y[i], 1.0 - x[i], 1.0 - y[i])
            if r[i] > d:
                max_violation = max(max_violation, r[i] - d)
        # Pairwise overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if r[i] + r[j] > dist:
                    max_violation = max(max_violation, r[i] + r[j] - dist)
        
        if max_violation <= 1e-9:
            break
        r -= max_violation
        r = np.clip(r, 0.0, None)
    
    # Final safety nudge: ensure strictly inside and non-negative
    x = np.clip(x, 1e-10, 1.0 - 1e-10)
    y = np.clip(y, 1e-10, 1.0 - 1e-10)
    r = np.clip(r, 0.0, None)
    
    return np.column_stack([x, y, r])


def _make_hex_initial():
    """Hexagonal packing initial guess."""
    pts = []
    # 5 columns, 5 rows, offset every other row
    for row in range(5):
        for col in range(5):
            x = 0.1 + col * 0.2 + (0.1 if row % 2 == 1 else 0.0)
            y = 0.1 + row * 0.2
            pts.append([x, y])
    # Add one more
    pts.append([0.5, 0.5])
    # Adjust to fit in [0,1]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Scale to fit
    scale = 0.8 / max(xmax - xmin, ymax - ymin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    for p in pts:
        p[0] = 0.5 + (p[0] - cx) * scale
        p[1] = 0.5 + (p[1] - cy) * scale

    # Estimate radius as half min distance
    min_dist = np.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            min_dist = min(min_dist, d)
    r0 = min_dist / 2 * 0.9
    # Also bound by distance to walls
    for i in range(len(pts)):
        r0 = min(r0, pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1])

    arr = np.array(pts)
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])


def _make_grid_initial():
    """Grid initial guess."""
    side = 6  # 6x6 = 36 points, enough for 26
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    pts = pts[:N]
    arr = np.array(pts)
    # Radius: half the cell size, with margin
    r0 = 0.5 / side * 0.9
    for i in range(N):
        r0 = min(r0, arr[i, 0], 1 - arr[i, 0], arr[i, 1], 1 - arr[i, 1])
    radii = np.full(N, r0)
    return np.column_stack([arr[:, 0], arr[:, 1], radii])


def _make_ring_initial():
    """Ring initial guess: center + concentric rings."""
    ring = np.zeros((N, 3))
    ring[0] = [0.5, 0.5, 0.12]
    idx = 1
    for radius, count in [(0.22, 6), (0.34, 8), (0.44, 11)]:
        for k in range(count):
            if idx >= N:
                break
            a = 2 * np.pi * k / count
            ring[idx] = [0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a), 0.05]
            idx += 1
        if idx >= N:
            break
    return ring


def _make_random_initial(seed):
    """Random initial guess."""
    rng = np.random.RandomState(seed)
    # Random positions in [0.1, 0.9]
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    # Estimate radius from nearest neighbor
    min_dists = np.full(N, 0.3)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dists[i] = min(min_dists[i], d)
    r = 0.4 * min_dists
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)))
    r = np.maximum(r, 0.01)
    return np.column_stack([x, y, r])


def _optimize_lbfgs(arr_init, lam_b, lam_o, maxiter=300):
    """Run L-BFGS-B optimization from initial guess."""
    v0 = arr_init.flatten()

    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

    res = minimize(
        _penalty_objective, v0,
        args=(lam_b, lam_o),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12}
    )

    return res.x.reshape(N, 3)


def _optimize_nelder_mead(arr_init, lam_b, lam_o, maxiter=200):
    """Run Nelder-Mead optimization from initial guess."""
    v0 = arr_init.flatten()

    res = minimize(
        _penalty_objective, v0,
        args=(lam_b, lam_o),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
    )

    # Clip to valid range
    x = np.clip(res.x[:N], 0, 1)
    y = np.clip(res.x[N:2*N], 0, 1)
    r = np.clip(res.x[2*N:3*N], 0, 0.5)

    return np.column_stack([x, y, r])


def construct():
    start_time = time.time()

    best_arr = None
    best_val = -np.inf

    # Generate initial guesses
    inits = []
    inits.append(('hex', _make_hex_initial()))
    inits.append(('grid', _make_grid_initial()))
    inits.append(('ring', _make_ring_initial()))

    # Several random inits
    for seed in [42, 123, 456, 789, 1024, 2048]:
        inits.append((f'rand_{seed}', _make_random_initial(seed)))

    # Lambda sets for different penalty strengths
    lambda_sets = [
        (50.0, 500.0),
        (100.0, 1000.0),
        (200.0, 2000.0),
    ]

    def _try_solution(arr):
        nonlocal best_arr, best_val
        # Fix feasibility
        arr_fixed = _fix_feasibility(arr)
        if _is_valid(arr_fixed):
            val = arr_fixed[:, 2].sum()
            if val > best_val:
                best_val = val
                best_arr = arr_fixed.copy()

    # Phase 1: L-BFGS-B with multiple initializations and lambda values
    for lam_b, lam_o in lambda_sets:
        if time.time() - start_time > DEADLINE - 20:
            break

        for name, arr_init in inits:
            if time.time() - start_time > DEADLINE - 10:
                break

            try:
                arr = _optimize_lbfgs(arr_init, lam_b, lam_o)
                _try_solution(arr)
            except Exception:
                continue

    # Phase 2: Nelder-Mead refinement on the best solution found so far
    if best_arr is not None:
        if time.time() - start_time < DEADLINE - 15:
            try:
                # Use higher penalty for refinement
                arr_nm = _optimize_nelder_mead(best_arr, 500.0, 5000.0, maxiter=150)
                _try_solution(arr_nm)
            except Exception:
                pass

        # Additional L-BFGS-B polish with high penalty
        if time.time() - start_time < DEADLINE - 10:
            try:
                arr_polish = _optimize_lbfgs(best_arr, 500.0, 5000.0, maxiter=200)
                _try_solution(arr_polish)
            except Exception:
                pass

    # If no valid solution found, try with looser validity
    if best_arr is None:
        # Try the grid as fallback
        best_arr = _make_grid_initial()
        best_val = best_arr[:, 2].sum()

    # Final check and cleanup
    if not _is_valid(best_arr):
        best_arr = _fix_feasibility(best_arr)

    if _is_valid(best_arr):
        return best_arr

    # Last resort: return grid
    return _make_grid_initial()
