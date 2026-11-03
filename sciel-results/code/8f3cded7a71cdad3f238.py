"""Pack 200 equal circles in a unit square to maximize common radius.

Uses a multi-stage strategy:
1. Generate multiple diverse initial configurations (hexagonal, random, grid).
2. For each initial configuration, perform gradient-based optimization using
   a soft-min objective with an increasing beta schedule.
3. Apply a final local refinement step using a direct force-directed method
   on the exact objective to squeeze out any remaining gains.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Compute the achievable radius for a set of points inside [-0.5, 0.5]^2."""
    if len(pts) < N:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check inside square
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0
    # Min distance to wall
    wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    # Min pairwise distance
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    r = min(wall, min_pair / 2.0)
    return max(r, 0.0)


def _generate_hexagonal_grid(n, jitter=0.0):
    """Generate a hexagonal (close-packing) grid arrangement for n points."""
    best_pts = None
    best_r = -1
    
    for R in range(10, 20):
        C = int(np.ceil((n + R // 2) / R))
        
        pts = []
        dx_x = 1.0 / (C - 0.5)
        dx_y = 1.0 / ((R - 1) * np.sqrt(3) / 2) if R > 1 else 1.0
        dx = min(dx_x, dx_y)
        dy = dx * np.sqrt(3) / 2
        
        start_x = -0.5 + (1.0 - (C - 0.5) * dx) / 2.0
        start_y = -0.5 + (1.0 - (R - 1) * dy) / 2.0
        
        for r in range(R):
            y = start_y + r * dy
            if r % 2 == 0:
                x_start = start_x
                n_pts = C
            else:
                x_start = start_x + dx / 2.0
                n_pts = C - 1
            
            for c in range(n_pts):
                x = x_start + c * dx
                if jitter > 0:
                    x += np.random.uniform(-jitter, jitter)
                    y += np.random.uniform(-jitter, jitter)
                pts.append([x, y])
                if len(pts) == n:
                    break
            if len(pts) == n:
                break
        
        if len(pts) == n:
            pts = np.array(pts)
            # Clip to square
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
            r_val = _compute_radius(pts)
            if r_val > best_r:
                best_r = r_val
                best_pts = pts
    
    if best_pts is None:
        # Fallback to simple grid
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:n]
        
    return best_pts


def _generate_random_grid(n, seed=None):
    """Generate a random grid-like configuration."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Use a distorted grid
    rows = 14
    cols = 15
    xs = -0.5 + (np.arange(cols) + 0.5) * (1.0 / cols)
    ys = -0.5 + (np.arange(rows) + 0.5) * (1.0 / rows)
    
    pts = []
    for y in ys:
        for x in xs:
            if len(pts) < n:
                # Add small random perturbation
                dx = rng.uniform(-0.01, 0.01)
                dy = rng.uniform(-0.01, 0.01)
                pts.append([x + dx, y + dy])
    
    pts = np.array(pts[:n])
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    return pts


def _generate_pure_random(n, seed=None):
    """Generate a purely random configuration."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    pts = rng.uniform(-0.45, 0.45, size=(n, 2))
    return pts


def _soft_min_obj_and_grad(x, beta):
    """
    Compute the soft-min objective (to be minimized) and its gradient.
    Objective: (1/beta) * log(sum(exp(-beta * c_i)))
    This is an approximation of -min(c_i). Minimizing this maximizes min(c_i).
    """
    pts = x.reshape(N, 2)
    n = N
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1e10)
    dist = np.sqrt(dist_sq)
    
    # Wall constraints
    wall_dists = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pairwise distances (upper triangle)
    iu = np.triu_indices(n, 1)
    pair_dists = dist[iu] / 2.0
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Calculate weights
    max_c = np.max(all_dists)
    exps = np.exp(-beta * (all_dists - max_c))
    sum_exps = np.sum(exps)
    
    # Objective value
    obj = -max_c + (1.0 / beta) * np.log(sum_exps)
    
    # Gradient
    grads = np.zeros_like(pts)
    
    # Pairwise gradients
    pair_weights = exps[:len(iu[0])] / sum_exps
    
    P = np.zeros((n, n))
    P[iu[0], iu[1]] = pair_weights
    P[iu[1], iu[0]] = pair_weights
    
    inv_dist_2 = 1.0 / (2.0 * dist)
    np.fill_diagonal(inv_dist_2, 0.0)
    
    V_x = diff[:, :, 0] * inv_dist_2
    V_y = diff[:, :, 1] * inv_dist_2
    
    grads[:, 0] = np.sum(P * V_x, axis=1)
    grads[:, 1] = np.sum(P * V_y, axis=1)
    
    # Wall gradients
    wall_weights = exps[len(iu[0]):] / sum_exps
    
    w_x_pos = wall_weights[0:n]
    w_x_neg = wall_weights[n:2*n]
    w_y_pos = wall_weights[2*n:3*n]
    w_y_neg = wall_weights[3*n:4*n]
    
    grads[:, 0] += w_x_pos - w_x_neg
    grads[:, 1] += w_y_pos - w_y_neg
    
    return obj, -grads


def _optimize_from_initial(pts_init, time_limit):
    """Optimize starting from an initial configuration."""
    t0 = time.time()
    x0 = pts_init.flatten().copy()
    best_pts = pts_init.copy()
    best_r = _compute_radius(pts_init)
    
    # Optimization
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - t0 > time_limit - 5:
            break
            
        try:
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            res = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x.reshape(N, 2)
            r_opt = _compute_radius(x_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = x_opt
                x0 = x_opt.copy()
                
        except Exception:
            pass
            
    return best_pts, best_r


def _force_refine(pts, time_limit):
    """
    Direct force-directed refinement on the exact objective.
    Try to increase the minimum distance by pushing points apart.
    """
    t0 = time.time()
    pts = pts.copy()
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # We'll use a simple iterative method:
    # 1. Find the closest pair or wall constraint.
    # 2. Move points to increase that distance.
    
    max_iters = 500
    step_size = 0.001
    
    for _ in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        # Compute distances
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        wall = np.min([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1]),
        ])
        
        min_pair = np.min(D)
        current_r = min(wall, min_pair / 2.0)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
        
        # If we're stuck, try a random perturbation
        # Find the active constraints
        if min_pair / 2.0 < wall:
            # Pairwise constraint is active
            i, j = np.unravel_index(np.argmin(D), D.shape)
            # Move i and j apart
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                direction = diff / dist
                pts[i] += direction * step_size * 0.5
                pts[j] -= direction * step_size * 0.5
        else:
            # Wall constraint is active
            # Find which wall
            min_x = np.min(pts[:, 0])
            max_x = np.max(pts[:, 0])
            min_y = np.min(pts[:, 1])
            max_y = np.max(pts[:, 1])
            
            d_left = min_x + 0.5
            d_right = 0.5 - max_x
            d_bottom = min_y + 0.5
            d_top = 0.5 - max_y
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            if min_d == d_left:
                idx = np.argmin(pts[:, 0])
                pts[idx, 0] += step_size
            elif min_d == d_right:
                idx = np.argmax(pts[:, 0])
                pts[idx, 0] -= step_size
            elif min_d == d_bottom:
                idx = np.argmin(pts[:, 1])
                pts[idx, 1] += step_size
            else:
                idx = np.argmax(pts[:, 1])
                pts[idx, 1] -= step_size
        
        # Clip to square
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Decay step size
        step_size *= 0.99
        
        if step_size < 1e-6:
            break
    
    return best_pts, best_r


def construct():
    t0 = time.time()
    
    # Generate multiple diverse initial configurations
    np.random.seed(42)
    
    initial_configs = [
        _generate_hexagonal_grid(N),
        _generate_hexagonal_grid(N, jitter=0.005),
        _generate_random_grid(N, seed=1),
        _generate_random_grid(N, seed=2),
        _generate_pure_random(N, seed=3),
        _generate_pure_random(N, seed=4),
    ]
    
    best_pts = None
    best_r = -1
    
    # Allocate time for each initial config
    num_configs = len(initial_configs)
    time_per_config = (_TIME_LIMIT - 10) / num_configs  # Leave 10s for final refinement
    
    for i, config in enumerate(initial_configs):
        remaining = _TIME_LIMIT - (time.time() - t0)
        if remaining < 5:
            break
        
        config_time = min(time_per_config, remaining - 2)
        
        try:
            pts_opt, r_opt = _optimize_from_initial(config, config_time)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt
        except Exception:
            pass
    
    # Final refinement with force-directed method
    if best_pts is not None:
        remaining = _TIME_LIMIT - (time.time() - t0)
        if remaining > 5:
            try:
                pts_final, r_final = _force_refine(best_pts, remaining - 2)
                if r_final > best_r:
                    best_r = r_final
                    best_pts = pts_final
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback
        best_pts = _generate_hexagonal_grid(N)
    
    return best_pts
