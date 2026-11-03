"""
Pack 200 equal circles in a unit square using L-BFGS-B optimization
with soft-min objective and force-directed pre-conditioning.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
_TIME_LIMIT = 135  # Leave margin for 150s kill

def _compute_radius_fast(pts):
    """Fast radius computation using cdist."""
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0
    
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if min_wall < 0:
        return 0.0

    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    We want to MAXIMIZE the soft-min of constraints, so MINIMIZE -soft_min.
    
    Constraints:
    - Wall distances: pts[:,0]+0.5, 0.5-pts[:,0], pts[:,1]+0.5, 0.5-pts[:,1]
    - Pair distances: |p_i - p_j| / 2 for all i<j
    """
    pts = x.reshape((N, 2))
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Build constraint array C
    C = np.empty(4*N + num_pairs)
    
    # Wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Pair values
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0
    
    # Soft min computation using log-sum-exp for numerical stability
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    weights = exp_shifted / sum_exp
    
    # Gradient computation
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients: d/d(pts) of wall distances
    # C[0:N] = pts[:,0] + 0.5, dC/dpts[:,0] = 1
    # C[N:2N] = 0.5 - pts[:,0], dC/dpts[:,0] = -1
    # C[2N:3N] = pts[:,1] + 0.5, dC/dpts[:,1] = 1
    # C[3N:4N] = 0.5 - pts[:,1], dC/dpts[:,1] = -1
    # grad of soft_min = -sum_k w_k * dC_k/dx / beta
    # obj = -soft_min, so grad_obj = sum_k w_k * dC_k/dx / beta
    
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    # d(dist/2)/d(pts[i,0]) = dx/(2*dist)
    # d(dist/2)/d(pts[j,0]) = -dx/(2*dist)
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _force_directed_precondition(pts, n_steps=30, dt=0.005):
    """Quick force-directed spreading to improve initial layout."""
    pts = pts.copy()
    n = len(pts)
    
    for step in range(n_steps):
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=0.1)
        
        forces = np.zeros_like(pts)
        
        for i, j in pairs:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist < 0.08:
                force_mag = (0.08 - dist) / 0.08
                forces[i] += (d / dist) * force_mag
                forces[j] -= (d / dist) * force_mag
        
        # Wall forces
        for axis in range(2):
            close_left = pts[:, axis] + 0.5 < 0.04
            if np.any(close_left):
                forces[close_left, axis] += 2.0 * (0.04 - (pts[close_left, axis] + 0.5))
            close_right = 0.5 - pts[:, axis] < 0.04
            if np.any(close_right):
                forces[close_right, axis] -= 2.0 * (0.04 - (0.5 - pts[close_right, axis]))
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-10:
            break
        actual_dt = dt / max(max_force, 1e-6)
        pts += actual_dt * forces
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def _hexagonal_initialization(n, jitter=0.0):
    """Generate a hexagonal layout with optional jitter."""
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    if jitter > 0:
        points += jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)
    
    return points

def _grid_initialization(n, jitter=0.0):
    """Generate a grid layout with optional jitter."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) < n:
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])
    
    grid = grid[:n]
    
    if jitter > 0:
        grid += jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid

def _run_lbfgs(pts, beta, time_limit, max_iter=500):
    """Run L-BFGS-B optimization with given beta."""
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def fun_and_grad(x):
        val, g = _soft_min_objective_and_grad(x, beta)
        return val, g
    
    try:
        res = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 20,
                'maxcor': 10
            }
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _final_polish(pts, time_limit):
    """High-precision polish with very high beta."""
    if time_limit < 5:
        return pts
    
    # Use very high beta for near-exact min
    beta = 500
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def fun_and_grad(x):
        val, g = _soft_min_objective_and_grad(x, beta)
        return val, g
    
    try:
        res = minimize(
            fun_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'maxfun': 5000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 20,
                'maxcor': 10
            }
        )
        new_pts = res.x.reshape(N, 2)
        if _compute_radius_fast(new_pts) >= _compute_radius_fast(pts):
            return new_pts
    except Exception:
        pass
    return pts

def _bottleneck_repair(coords, time_limit=5.0):
    """Targeted repair of the most constraining pairs and wall violations."""
    coords = coords.copy()
    n = len(coords)
    start_time = time.time()
    
    for it in range(100):
        if time.time() - start_time > time_limit:
            break
        
        r = _compute_radius_fast(coords)
        if r <= 0:
            break
        
        D = cdist(coords, coords)
        np.fill_diagonal(D, np.inf)
        min_dist = np.min(D)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        wall_dists = np.array([
            coords[:, 0] + 0.5,
            0.5 - coords[:, 0],
            coords[:, 1] + 0.5,
            0.5 - coords[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        if min_wall < min_dist / 2.0:
            min_idx = np.argmin(wall_dists)
            for axis in range(2):
                for side in range(2):
                    if side == 0:
                        d = coords[min_idx, axis] + 0.5
                    else:
                        d = 0.5 - coords[min_idx, axis]
                    if np.isclose(d, min_wall, atol=1e-12):
                        move = 0.03 * r
                        if side == 0:
                            coords[min_idx, axis] += move
                        else:
                            coords[min_idx, axis] -= move
                        coords[min_idx, axis] = np.clip(coords[min_idx, axis], -0.5, 0.5)
                        break
        else:
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-12:
                move = 0.03 * r
                nx, ny = dx / dist, dy / dist
                
                new_i = coords[i].copy()
                new_i[0] += nx * move
                new_i[1] += ny * move
                new_i = np.clip(new_i, -0.5, 0.5)
                coords[i] = new_i
                new_r = _compute_radius_fast(coords)
                if new_r <= r:
                    coords[i] = coords[i] - np.array([nx * move, ny * move])
                
                new_j = coords[j].copy()
                new_j[0] -= nx * move
                new_j[1] -= ny * move
                new_j = np.clip(new_j, -0.5, 0.5)
                coords[j] = new_j
                new_r2 = _compute_radius_fast(coords)
                if new_r2 <= r:
                    coords[j] = coords[j] + np.array([nx * move, ny * move])
    
    return coords

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Multiple diverse starts
    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.01)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.01)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    for name, init_func in initializers:
        if time.time() > deadline - 30:
            break
        
        try:
            coords = init_func()
            
            # Quick force-directed preconditioning
            if time.time() < deadline - 25:
                coords = _force_directed_precondition(coords, n_steps=20, dt=0.003)
            
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            # L-BFGS-B optimization with progressive beta
            if time.time() < deadline - 20:
                remaining = deadline - time.time() - 10
                
                # Progressive beta schedule: start smooth, get sharper
                betas = [10, 50, 200]
                current_pts = coords
                
                for beta in betas:
                    if time.time() > deadline - 15:
                        break
                    
                    beta_time = min(remaining / 3, 30)
                    if beta_time < 3:
                        break
                    
                    optimized = _run_lbfgs(current_pts, beta, time_limit=beta_time, max_iter=300)
                    r_opt = _compute_radius_fast(optimized)
                    
                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_coords = optimized.copy()
                    
                    if r_opt >= r:
                        current_pts = optimized
                    else:
                        break
                    
                    remaining = deadline - time.time() - 10
        
        except Exception:
            continue
    
    # Final high-precision polish on the best configuration
    if best_coords is not None and time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            polished = _final_polish(best_coords, time_limit=remaining)
            r_polished = _compute_radius_fast(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_coords = polished.copy()
    
    # Final bottleneck repair
    if best_coords is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            repaired = _bottleneck_repair(best_coords, time_limit=remaining)
            r_repaired = _compute_radius_fast(repaired)
            if r_repaired > best_radius:
                best_radius = r_repaired
                best_coords = repaired.copy()
    
    # Final validation
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)
    
    return best_coords

# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)

def objective(coords):
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    return _hexagonal_initialization(n)

# EVOLVE-BLOCK-END
