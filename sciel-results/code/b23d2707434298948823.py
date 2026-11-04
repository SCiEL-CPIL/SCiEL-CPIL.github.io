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


def _soft_objective(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)

    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Boundary penalty gradient
    # d(bpen)/dxi = 2*bv * (db_left/dxi + db_right/dxi)
    # db_left/dxi = -1 if b_left > 0 else 0
    # db_right/dxi = +1 if b_right > 0 else 0
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    # For each pair (i,j) with ov > 0:
    # d(ov)/dxi_i = -(xi-xj)/dist, d(ov)/dxi_j = +(xi-xj)/dist
    # d(ov)/dri = 1, d(ov)/drj = 1
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        # Handle degenerate case
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        # Accumulate gradients using np.add.at
        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

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


def _project_feasible(x, n=N, max_iter=500, tol=1e-10):
    """
    Project a configuration to feasibility by iteratively resolving violations.
    Reduces radii and adjusts positions to eliminate boundary/overlap violations.
    Vectorized overlap resolution.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
        
        # Fix boundary violations: reduce radius to fit
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        too_big = rs > min_d
        if np.any(too_big):
            rs[too_big] = min_d[too_big] * (1 - 1e-9)
            pts[:, 2] = rs
        
        # Fix overlap violations: reduce radii
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)
        
        if not np.any(ov > tol):
            break
        
        # Reduce radii proportionally to resolve overlaps (vectorized)
        # Each circle's total overlap contribution
        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)
        
        # Reduce each radius by half its total overlap
        reduction = overlap_per_circle / 2.0 + tol
        rs = np.maximum(0.01, rs - reduction)
        pts[:, 2] = rs
    
    return pts.ravel()


def _grow_radii(x, n=N, max_iter=200, step=0.001, tol=1e-8):
    """
    Try to grow radii while maintaining feasibility.
    Uses a greedy approach: try to increase each radius slightly,
    check feasibility, and keep the increase if valid.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        if _remaining() < 2.0:
            break
        
        improved = False
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # For each circle, compute max possible radius increase
        # limited by boundary and overlaps
        for i in range(n):
            # Boundary limit
            d_left = xs[i]
            d_bot = ys[i]
            d_right = 1.0 - xs[i]
            d_top = 1.0 - ys[i]
            max_r_boundary = min(d_left, d_bot, d_right, d_top)
            
            # Overlap limits with other circles
            max_r_overlap = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                # r_i + r_j <= dist => r_i <= dist - r_j
                limit = dist - rs[j]
                if limit < max_r_overlap:
                    max_r_overlap = limit
            
            # Try to grow
            new_r = min(max_r_overlap, rs[i] + step)
            if new_r > rs[i] + 1e-12:
                old_r = rs[i]
                rs[i] = new_r
                pts[i, 2] = new_r
                
                # Check if still valid
                if _is_valid(pts.ravel(), n, tol=tol):
                    improved = True
                else:
                    rs[i] = old_r
                    pts[i, 2] = old_r
        
        if not improved:
            break
    
    return pts.ravel()


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    # For 26 circles, use a hexagonal-like arrangement
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est
    
    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        start_x = (1.0 - total_width) / 2.0
        for j in range(count):
            x = start_x + j * x_spacing
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    
    # Scale to fit
    xs = np.array(xs)
    ys = np.array(ys)
    rs = np.array(rs)
    
    # Center and scale
    xs = (xs - xs.min()) / (xs.max() - xs.min())
    ys = (ys - ys.min()) / (ys.max() - ys.min())
    
    # Ensure they fit in [0,1] with some margin
    margin = 0.05
    xs = xs * (1.0 - 2 * margin) + margin
    ys = ys * (1.0 - 2 * margin) + margin
    
    # Recalculate radii based on actual spacing
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    
    min_dist = np.full(n, 1.0)
    np.minimum.at(min_dist, i_idx, dist)
    np.minimum.at(min_dist, j_idx, dist)
    
    # Also consider walls
    wall_dist = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
    min_dist = np.minimum(min_dist, wall_dist)
    
    rs = 0.5 * min_dist * 0.95
    
    x = np.empty(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def _random_init(rng, n=N):
    """Random initial guess."""
    xs = rng.uniform(0.2, 0.8, n)
    ys = rng.uniform(0.2, 0.8, n)
    rs = rng.uniform(0.05, 0.1, n)
    x = np.empty(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def construct():
    global _START
    _START = time.time()
    
    best_val = -np.inf
    best_x = None
    
    bounds = [(0, 1), (0, 1), (1e-6, 1)] * N
    
    # Initial guesses
    rng = np.random.default_rng(42)
    inits = [_hex_packing_init(N)]
    for _ in range(10):
        inits.append(_random_init(rng, N))
    
    # Annealing schedule for mu
    mus = [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1]
    
    for init in inits:
        if _remaining() < 10.0:
            break
        
        x = init.copy()
        # Project to feasible start
        x = _project_feasible(x, N)
        
        for mu in mus:
            if _remaining() < 5.0:
                break
            
            # Optimize
            res = minimize(_soft_objective, x, jac=True, method='L-BFGS-B',
                           bounds=bounds, args=(mu,),
                           options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
            x = res.x
            
            # Project to feasible
            x = _project_feasible(x, N)
            
            # Grow radii
            if _remaining() > 5.0:
                x = _grow_radii(x, N, max_iter=50)
                x = _project_feasible(x, N)
            
            val = _score(x, N)
            if val > best_val:
                best_val = val
                best_x = x.copy()
    
    # Final polish
    if best_x is not None and _remaining() > 10.0:
        # Very low mu for fine tuning
        for mu in [0.01, 0.001]:
            if _remaining() < 5.0:
                break
            res = minimize(_soft_objective, best_x, jac=True, method='L-BFGS-B',
                           bounds=bounds, args=(mu,),
                           options={'maxiter': 500, 'gtol': 1e-9, 'ftol': 1e-12})
            x = res.x
            x = _project_feasible(x, N)
            x = _grow_radii(x, N, max_iter=100)
            x = _project_feasible(x, N)
            val = _score(x, N)
            if val > best_val:
                best_val = val
                best_x = x.copy()
    
    # Fallback
    if best_x is None:
        x = _hex_packing_init(N)
        x = _project_feasible(x, N)
        best_x = x
        best_val = _score(x, N)
    
    # Return as (n, 3) array
    return best_x.reshape(N, 3)


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _is_valid(res.ravel(), N))
