"""
Combined strategy: Binary search on radius with L-BFGS-B optimization using smooth violation objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5

# EVOLVE-BLOCK-START

def _score(c):
    """
    Compute the largest common radius for centres c (N,2).
    Returns 0.0 if invalid.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    # Wall distances
    min_bound = np.min(HALF - np.abs(c))
    
    return min(min_pair, min_bound)


def _hex_seed(r):
    """
    Generate a hexagonal lattice seed with spacing approx 2r.
    """
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x_offset = 0.5 * dx if (row % 2) else 0.0
        x = -HALF + r + x_offset
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
        
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """
    Generate a simple grid seed.
    """
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _violation_and_grad(x, r):
    """
    Compute violation and gradient for L-BFGS-B.
    Violation = sum of max(0, 2r - dist_ij) for all pairs + wall violations.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    
    # Pair violations: max(0, 2r - dist)
    pair_viol = np.maximum(0, 2 * r - dist)
    
    # Gradient from pairs
    weight = np.where((dist < 2 * r) & (dist > 1e-12), (2 * r - dist) / dist, 0.0)
    grad_pairs = np.sum(diff * weight[:, :, None], axis=1)
    
    # Wall violations
    wall_viol = np.zeros(N)
    grad_walls = np.zeros_like(pts)
    
    too_left_x = pts[:, 0] < -HALF + r
    if np.any(too_left_x):
        wall_viol[too_left_x] += (-HALF + r - pts[too_left_x, 0])
        grad_walls[too_left_x, 0] -= 1.0
    
    too_right_x = pts[:, 0] > HALF - r
    if np.any(too_right_x):
        wall_viol[too_right_x] += (pts[too_right_x, 0] - (HALF - r))
        grad_walls[too_right_x, 0] += 1.0
    
    too_low_y = pts[:, 1] < -HALF + r
    if np.any(too_low_y):
        wall_viol[too_low_y] += (-HALF + r - pts[too_low_y, 1])
        grad_walls[too_low_y, 1] -= 1.0
    
    too_high_y = pts[:, 1] > HALF - r
    if np.any(too_high_y):
        wall_viol[too_high_y] += (pts[too_high_y, 1] - (HALF - r))
        grad_walls[too_high_y, 1] += 1.0
    
    total_viol = np.sum(pair_viol) + np.sum(wall_viol)
    total_grad = grad_pairs + grad_walls
    
    return total_viol, total_grad.ravel()


def _lbfgs_feasible(pts, r, deadline=None, max_iter=200):
    """
    Try to find a feasible configuration for radius r using L-BFGS-B.
    """
    pts = pts.astype(np.float64).copy()
    x0 = pts.ravel()
    
    bounds = [(-HALF + r, HALF - r)] * (2 * N)
    
    def obj(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        v, _ = _violation_and_grad(x, r)
        return v
    
    def grad(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros(2 * N)
        _, g = _violation_and_grad(x, r)
        return g
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', jac=grad,
            bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pts = result.x.reshape(N, 2)
    except Exception:
        pass
    
    # Check feasibility
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    
    success = (min_pair >= 2 * r - 1e-8) and (min_wall >= r - 1e-8)
    return success, pts


def _pgd_feasible(pts, r, max_iter=500, deadline=None):
    """
    Fallback: Projected Gradient Descent for feasibility.
    """
    pts = pts.astype(np.float64).copy()
    
    low_bound = -HALF + r
    high_bound = HALF - r
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        overlap = (dist < 2 * r) & (dist > 1e-12)
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)
        
        wall_grad = np.zeros_like(pts)
        tl = pts[:, 0] < low_bound
        tr = pts[:, 0] > high_bound
        tb = pts[:, 1] < low_bound
        tt = pts[:, 1] > high_bound
        if np.any(tl): wall_grad[tl, 0] += (low_bound - pts[tl, 0])
        if np.any(tr): wall_grad[tr, 0] -= (pts[tr, 0] - high_bound)
        if np.any(tb): wall_grad[tb, 1] += (low_bound - pts[tb, 1])
        if np.any(tt): wall_grad[tt, 1] -= (pts[tt, 1] - high_bound)
        
        grad += wall_grad
        
        step = 0.1
        new_pts = pts + step * grad
        new_pts[:, 0] = np.clip(new_pts[:, 0], low_bound, high_bound)
        new_pts[:, 1] = np.clip(new_pts[:, 1], low_bound, high_bound)
        
        diff_new = new_pts[:, None, :] - new_pts[None, :, :]
        dist_new = np.sqrt(np.sum(diff_new * diff_new, axis=2))
        np.fill_diagonal(dist_new, np.inf)
        min_pair = np.min(dist_new)
        min_wall = np.min(HALF - np.abs(new_pts))
        
        if min_pair >= 2 * r - 1e-9 and min_wall >= r - 1e-9:
            return True, new_pts
            
        pts = new_pts
    
    diff_final = pts[:, None, :] - pts[None, :, :]
    dist_final = np.sqrt(np.sum(diff_final * diff_final, axis=2))
    np.fill_diagonal(dist_final, np.inf)
    min_pair = np.min(dist_final)
    min_wall = np.min(HALF - np.abs(pts))
    
    success = (min_pair >= 2 * r - 1e-9) and (min_wall >= r - 1e-9)
    return success, pts


def _polish(pts, deadline):
    """
    High-precision polish pass to extract maximum radius.
    """
    current_r = _score(pts)
    
    for _ in range(8):
        if time.time() > deadline - 3:
            break
        target_r = current_r + 0.00005
        ok, polished = _lbfgs_feasible(pts, target_r, deadline=deadline, max_iter=150)
        if ok:
            new_r = _score(polished)
            if new_r > current_r + 1e-9:
                pts = polished
                current_r = new_r
            else:
                break
        else:
            break
    
    return pts, current_r


def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    r_low = 0.068
    r_high = 0.073
    
    best_pts = None
    best_r = 0.0
    
    init_pts = _hex_seed(r_low)
    ok, init_pts = _lbfgs_feasible(init_pts, r_low, deadline=deadline, max_iter=150)
    if not ok:
        ok, init_pts = _pgd_feasible(init_pts, r_low, max_iter=300, deadline=deadline)
    if ok:
        best_pts = init_pts
        best_r = _score(best_pts)
        r_low = best_r
    
    for _ in range(20):
        if time.time() > deadline - 15:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        found = False
        candidate_pts = None
        
        seeds = [
            _hex_seed(r_mid),
            _grid_seed(r_mid),
            _hex_seed(r_mid) + np.random.randn(N, 2) * 0.003,
            _grid_seed(r_mid) + np.random.randn(N, 2) * 0.003
        ]
        
        for seed in seeds:
            if time.time() > deadline - 15:
                break
                
            seed = np.clip(seed, -HALF + r_mid, HALF - r_mid)
            
            ok, pts = _lbfgs_feasible(seed, r_mid, deadline=deadline, max_iter=200)
            if not ok:
                ok, pts = _pgd_feasible(seed, r_mid, max_iter=300, deadline=deadline)
            if ok:
                found = True
                candidate_pts = pts
                break
        
        if found:
            r_low = r_mid
            best_pts = candidate_pts
            best_r = _score(best_pts)
            if best_r > r_low:
                r_low = best_r
        else:
            r_high = r_mid
            
        if r_high - r_low < 1e-5:
            break
    
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_r = _polish(best_pts, deadline)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
        
    return best_pts

# EVOLVE-BLOCK-END
