"""
Pack 200 equal circles in a unit square using gradient-based optimization
with a soft-min objective to handle the max-min constraint.
Combines the current L-BFGS-B soft-min approach with multi-start diversity
and a final high-precision polish pass.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import cKDTree

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s limit

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
        return 0.0
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    # Use pdist for exact min pairwise distance
    min_pair = np.min(pdist(pts)) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min(vals, beta):
    """
    Compute the soft minimum of a set of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective function for minimization.
    We want to MAXIMIZE the soft-min of constraints.
    So we MINIMIZE -soft_min.
    
    x: flattened array of shape (2*N,)
    beta: sharpness parameter for soft-min
    """
    pts = x.reshape((N, 2))
    
    # Wall terms
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair terms
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    D = cdist(pts, pts)
    pair_dists = D[i_idx, j_idx]
    
    C = np.empty(4*N + num_pairs)
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    C[4*N:] = pair_dists / 2.0
    
    # Compute soft min and weights
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    weights = exp_shifted / sum_exp
    
    # Compute Gradient
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)
    
    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top
    
    # Pair gradients
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    pair_weights = weights[4*N:]
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                continue
        
        r_actual = _get_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _gradient_refine_kdtree(centers, max_iter=5000, lr=0.003, k_neighbors=8):
    """
    Gradient-based refinement using KDTree for neighbor queries.
    """
    centers = centers.copy()
    best_t = _get_radius(centers)
    best_centers = centers.copy()
    
    no_improve_count = 0
    current_lr = lr
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        
        d = pdist(centers)
        min_pair_dist = np.min(d)
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall)
        
        if current_r > best_t + 1e-12:
            best_t = current_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        tree = cKDTree(centers)
        k = min(k_neighbors, N)
        dists, idx = tree.query(centers, k=k)
        
        grad = np.zeros_like(centers)
        
        for i in range(N):
            for k_idx in range(1, k):
                j = idx[i, k_idx]
                diff = centers[i] - centers[j]
                dist = dists[i, k_idx]
                if dist > 1e-10:
                    weight = 1.0 / (dist * dist + 1e-8)
                    grad[i] += weight * diff
            
            for coord in range(2):
                if centers[i, coord] < 0:
                    wall_dist = 0.5 + centers[i, coord]
                    if wall_dist < 0.2:
                        grad[i, coord] += 1.0 / (wall_dist + 1e-6)
                else:
                    wall_dist = 0.5 - centers[i, coord]
                    if wall_dist < 0.2:
                        grad[i, coord] -= 1.0 / (wall_dist + 1e-6)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        centers += current_lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        if no_improve_count > 100:
            current_lr *= 0.85
            no_improve_count = 0
            if current_lr < 1e-9:
                break
        elif no_improve_count == 30:
            current_lr *= 0.95
    
    return best_centers, best_t

def _run_l_bfgs_optimization(pts, time_limit, betas=None):
    """Run L-BFGS-B optimization with progressively increasing beta."""
    if betas is None:
        betas = [5, 10, 20, 50, 100, 200]
        
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            maxiter = 200 if beta <= 50 else 100
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
            
    return best_r, best_x.reshape((N, 2))

def _high_precision_polish(pts, time_limit):
    """Final high-precision polish with very high beta."""
    betas_polish = [300, 500, 1000, 2000]
    return _run_l_bfgs_optimization(pts, time_limit, betas=betas_polish)

def _force_directed_phase(pts, time_limit, target_r=None):
    """
    Force-directed relaxation: repel overlapping circles, attract to fill space.
    """
    if target_r is None:
        target_r = _get_radius(pts)
    
    t_start = time.time()
    pts = pts.copy()
    velocities = np.zeros_like(pts)
    damping = 0.9
    dt = 0.001
    repulsion_strength = 1.0
    attraction_strength = 0.1
    
    target_dist = 2 * target_r * 1.05
    
    for step in range(5000):
        if time.time() - t_start > time_limit:
            break
        
        forces = np.zeros_like(pts)
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        close_mask = D < 3 * target_dist
        close_mask = np.triu(close_mask, k=1)
        
        i_idx, j_idx = np.where(close_mask)
        
        for i, j in zip(i_idx, j_idx):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                diff = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(diff)
            
            if dist < target_dist:
                force_mag = repulsion_strength * (target_dist - dist) / target_dist
                force_dir = diff / dist
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
        
        wall_margin = 0.02
        for i in range(N):
            if pts[i, 0] < -0.5 + wall_margin:
                forces[i, 0] += attraction_strength * (-0.5 + wall_margin - pts[i, 0])
            if pts[i, 0] > 0.5 - wall_margin:
                forces[i, 0] -= attraction_strength * (pts[i, 0] - (0.5 - wall_margin))
            if pts[i, 1] < -0.5 + wall_margin:
                forces[i, 1] += attraction_strength * (-0.5 + wall_margin - pts[i, 1])
            if pts[i, 1] > 0.5 - wall_margin:
                forces[i, 1] -= attraction_strength * (pts[i, 1] - (0.5 - wall_margin))
        
        velocities = damping * velocities + dt * forces
        pts += dt * velocities
        pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        if step > 100 and np.max(np.abs(velocities)) < 1e-8:
            break
    
    return pts

_START_TIME = time.time()

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    global _START_TIME
    _START_TIME = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Generate multiple diverse initial configurations
    base_centers = _generate_hex_grid()
    
    centers_list = [base_centers.copy()]
    
    # Add perturbed versions for diversity
    rng = np.random.RandomState(42)
    for _ in range(4):
        perturbed = base_centers.copy()
        noise = rng.normal(0, 0.008, size=perturbed.shape)
        perturbed += noise
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.45, 0.45)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.45, 0.45)
        centers_list.append(perturbed)
    
    # Phase 1: Force-directed relaxation from best hex initial
    elapsed = time.time() - _START_TIME
    remaining = _TIME_LIMIT - elapsed - 60
    if remaining > 15:
        pts_fd = _force_directed_phase(base_centers, time_limit=min(20, remaining))
        r_fd = _get_radius(pts_fd)
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd
    
    # Phase 2: Multi-start L-BFGS optimization
    elapsed = time.time() - _START_TIME
    remaining = _TIME_LIMIT - elapsed - 30
    if remaining > 20:
        time_per_start = max(8.0, remaining / len(centers_list))
        
        for centers in centers_list:
            if time.time() - _START_TIME > _TIME_LIMIT - 30:
                break
            
            start_time = time.time()
            r_opt, pts_opt = _run_l_bfgs_optimization(centers, time_limit=time_per_start)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = pts_opt.copy()
    
    # Phase 3: KDTree-based gradient refinement on best result
    if best_pts is not None:
        elapsed = time.time() - _START_TIME
        remaining = _TIME_LIMIT - elapsed - 20
        if remaining > 10:
            pts_refined, r_refined = _gradient_refine_kdtree(
                best_pts, max_iter=3000, lr=0.002, k_neighbors=10
            )
            if r_refined > best_r:
                best_r = r_refined
                best_pts = pts_refined
    
    # Phase 4: High-precision polish with very high beta
    if best_pts is not None:
        elapsed = time.time() - _START_TIME
        remaining = _TIME_LIMIT - elapsed - 3
        if remaining > 5:
            r_polish, pts_polish = _high_precision_polish(best_pts, time_limit=remaining)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Final safety: clip to ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
