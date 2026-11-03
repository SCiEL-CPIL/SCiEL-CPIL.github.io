"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# Deadline guard: 140s wall-clock (10s safety margin under 150s)
_DEADLINE = 140.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy_and_grad(x, n=N):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + penalty for violations.
    Violations: circles outside [0,1]^2, or overlapping.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Penalty for boundary violations
    # Each circle center must be in [r, 1-r]
    # boundary violation: max(0, r - x, r - y, r - (1-x), r - (1-y))
    # We use a smooth penalty: sum of max(0, violation)^2
    # For efficiency, compute violations
    # x - r < 0 -> violation
    # 1 - x - r < 0 -> violation
    # y - r < 0
    # 1 - y - r < 0

    # Boundary violations
    bv = np.maximum(0, rs - xs) + np.maximum(0, rs - ys) + \
         np.maximum(0, rs - (1.0 - xs)) + np.maximum(0, rs - (1.0 - ys))
    bpen = np.sum(bv ** 2)

    # Overlap violations: for each pair i,j, dist < ri + rj
    # violation_ij = max(0, ri + rj - dist_ij)
    # We need to compute all pairs
    # For n=26, pairs = 325, manageable
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open = np.sum(ov ** 2)

    total_penalty = bpen + open
    energy = obj + 1000.0 * total_penalty

    # Gradient
    grad = np.zeros_like(x)

    # Gradient of obj: d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Gradient of boundary penalty
    # For circle i:
    # bv_i = max(0, ri - xi) + max(0, ri - yi) + max(0, ri - (1-xi)) + max(0, ri - (1-yi))
    # d(bv_i)/dxi = -1 if ri > xi else 0, +1 if ri > (1-xi) else 0
    # d(bv_i)/dyi = -1 if ri > yi else 0, +1 if ri > (1-yi) else 0
    # d(bv_i)/dri = 1 if ri > xi else 0, 1 if ri > yi else 0, 1 if ri > (1-xi) else 0, 1 if ri > (1-yi) else 0
    # d(bpen)/dxi = 2 * bv_i * d(bv_i)/dxi
    # etc.

    # Compute derivatives for each circle
    # This is a bit tedious but n=26 is small
    for i in range(n):
        xi, yi, ri = xs[i], ys[i], rs[i]
        # bv components
        b1 = max(0, ri - xi)  # left
        b2 = max(0, ri - yi)  # bottom
        b3 = max(0, ri - (1.0 - xi))  # right
        b4 = max(0, ri - (1.0 - yi))  # top
        bvi = b1 + b2 + b3 + b4

        # db/dxi
        db_dxi = 0.0
        if b1 > 0: db_dxi += -1.0
        if b3 > 0: db_dxi += 1.0
        # db/dyi
        db_dyi = 0.0
        if b2 > 0: db_dyi += -1.0
        if b4 > 0: db_dyi += 1.0
        # db/dri
        db_dri = 0.0
        if b1 > 0: db_dri += 1.0
        if b2 > 0: db_dri += 1.0
        if b3 > 0: db_dri += 1.0
        if b4 > 0: db_dri += 1.0

        # d(bpen)/dxi = 2 * bvi * db_dxi
        grad[3 * i] += 1000.0 * 2 * bvi * db_dxi
        grad[3 * i + 1] += 1000.0 * 2 * bvi * db_dyi
        grad[3 * i + 2] += 1000.0 * 2 * bvi * db_dri

    # Gradient of overlap penalty
    # For each pair (i,j), ov_ij = max(0, ri+rj - dist_ij)
    # d(ov_ij)/dxi = -(xi-xj)/dist * 1 if ov>0 else 0
    # d(ov_ij)/dri = 1 if ov>0 else 0
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        if ov[k] > 0:
            dist_k = dist[k]
            if dist_k < 1e-12:
                # Degenerate: push apart
                dxi = 1.0
                dyi = 0.0
            else:
                dxi = (xs[i] - xs[j]) / dist_k
                dyi = (ys[i] - ys[j]) / dist_k
            # d(open)/dxi_i = 2*ov * (-dxi)
            grad[3 * i] += 1000.0 * 2 * ov[k] * (-dxi)
            grad[3 * i + 1] += 1000.0 * 2 * ov[k] * (-dyi)
            grad[3 * i + 2] += 1000.0 * 2 * ov[k] * 1.0
            # d(open)/dxi_j = 2*ov * (dxi)
            grad[3 * j] += 1000.0 * 2 * ov[k] * dxi
            grad[3 * j + 1] += 1000.0 * 2 * ov[k] * dyi
            grad[3 * j + 2] += 1000.0 * 2 * ov[k] * 1.0

    return energy, grad


def _is_valid(x, n=N, tol=1e-6):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    # Boundary
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    # Overlap
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    # For 26 circles, use a hexagonal-like arrangement
    # Approximate: 5 rows, varying counts
    # Row counts: 5, 5, 5, 5, 5, 1? No, 26 = 5+5+5+5+5+1? 
    # Let's do 6 rows: 5,5,5,5,4,2? 
    # Simpler: use a grid with slight perturbation
    # Or use known good hex packing
    # For now, use a 5x5 grid + 1, with hex offset
    rows = []
    # Try to fit 26 circles
    # Hexagonal: row i has k or k+1 circles
    # Let's use 5 rows of 5 and 1 extra, or 6 rows
    # Actually, let's use a simple approach: 5 rows, counts [5,5,5,5,5,1] won't work well
    # Use 6 rows: [5,5,5,5,4,2] = 26
    row_counts = [5, 5, 5, 5, 4, 2]
    # Total = 26
    # Row height in hex: for circle radius r, vertical spacing = r*sqrt(3)
    # We'll estimate radius
    # Total width ~ 5*2r = 10r <= 1 -> r <= 0.1
    # Total height ~ 6*2r*0.866 = 10.39r <= 1 -> r <= 0.096
    # So r ~ 0.09
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est
    
    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        # Center the row
        total_width = (count - 1) * x_spacing
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * x_spacing
            # Alternate rows offset
            if i % 2 == 1:
                x += x_spacing / 2
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    
    # Scale to fit in [0,1]^2
    # Compute bounding box
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    rs_arr = np.array(rs)
    
    # Find max extent
    min_x = np.min(xs_arr - rs_arr)
    max_x = np.max(xs_arr + rs_arr)
    min_y = np.min(ys_arr - rs_arr)
    max_y = np.max(ys_arr + rs_arr)
    
    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)
    
    # Scale positions and radii
    xs_arr = (xs_arr - min_x) * scale
    ys_arr = (ys_arr - min_y) * scale
    rs_arr = rs_arr * scale
    
    # Center in [0,1]^2
    xs_arr += (1.0 - width * scale) / 2
    ys_arr += (1.0 - height * scale) / 2
    
    x = np.zeros(3 * n)
    x[0::3] = xs_arr
    x[1::3] = ys_arr
    x[2::3] = rs_arr
    
    return x


def _random_init(n=N, seed=None):
    """Random initial guess."""
    rng = np.random.default_rng(seed)
    # Random positions in [0.1, 0.9]
    xs = rng.uniform(0.1, 0.9, n)
    ys = rng.uniform(0.1, 0.9, n)
    rs = rng.uniform(0.05, 0.12, n)
    x = np.zeros(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def construct():
    global _START
    _START = time.time()
    
    best_x = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = []
    
    # Hex packing
    inits.append(_hex_packing_init(N))
    
    # Random inits
    for seed in [42, 123, 456, 789, 1000]:
        inits.append(_random_init(N, seed=seed))
    
    # Grid init
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    grid_x = np.zeros(3 * N)
    grid_x[0::3] = [c[0] for c in cells[:N]]
    grid_x[1::3] = [c[1] for c in cells[:N]]
    grid_x[2::3] = [c[2] for c in cells[:N]]
    inits.append(grid_x)
    
    for init_x in inits:
        if _remaining() < 15.0:
            break
        
        # Optimize
        try:
            res = minimize(
                _energy_and_grad,
                init_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x_opt = res.x
            score = _score(x_opt, N)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Polish: if we have a valid solution, do a final high-precision pass
    if best_x is not None and _remaining() > 10.0:
        try:
            res = minimize(
                _energy_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            x_opt = res.x
            score = _score(x_opt, N)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # If still no valid solution, fall back to grid
    if best_x is None or not _is_valid(best_x, N):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_x = np.array(cells[:N]).ravel()
    
    return best_x.reshape(N, 3)
