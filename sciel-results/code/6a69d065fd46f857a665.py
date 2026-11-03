"""Packing 50 equal circles in a unit square [-0.5, 0.5]^2.
Uses a multi-stage approach:
1. Fast force-relaxation to get a reasonable starting point.
2. Langevin dynamics with a time-dependent temperature schedule.
3. Soft-min radius optimization with gradient descent.
4. Local refinement with coordinate descent.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_FRAC = 0.92  # finish by 92% of 150s = 138s


def _radius(centers):
    """Compute the valid radius for a set of centers in [-0.5,0.5]^2."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    wall = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)


def _radius_fast(centers):
    """Faster radius using vectorized computation."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    wall = np.minimum(0.5 - np.abs(centers), axis=1)
    min_wall = wall.min()
    return min(min_pair / 2.0, min_wall)


def _soft_min_radius(centers, beta=50.0):
    """Compute a soft-min approximation of the radius, differentiable."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Soft-min of pairwise distances / 2
    # Use log-sum-exp trick for numerical stability
    dist = np.sqrt(dist_sq + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Soft min: -log(sum(exp(-beta * d))) / beta
    # But we want the min of d/2, so:
    # soft_min = -1/beta * log(sum(exp(-beta * d / 2)))
    # To avoid overflow, subtract the min:
    vals = dist / 2.0
    vals_flat = vals[~np.isinf(vals)]
    if len(vals_flat) == 0:
        pair_val = 0.0
    else:
        v_min = vals_flat.min()
        # Only compute over finite values
        finite_mask = ~np.isinf(vals)
        exp_vals = np.exp(-beta * (vals - v_min))
        exp_vals[~finite_mask] = 0.0
        pair_val = v_min - np.log(exp_vals.sum()) / beta
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    w_min = wall_dists.min()
    exp_wall = np.exp(-beta * (wall_dists - w_min))
    wall_val = w_min - np.log(exp_wall.sum()) / beta
    
    return min(pair_val, wall_val)


def _soft_min_radius_and_grad(centers, beta=50.0):
    """Compute soft-min radius and its gradient."""
    centers = np.array(centers, dtype=np.float64)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    dist = np.sqrt(dist_sq + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Soft min of d/2
    vals = dist / 2.0
    finite_mask = ~np.isinf(vals)
    
    if np.any(finite_mask):
        v_min = vals[finite_mask].min()
        exp_vals = np.exp(-beta * (vals - v_min))
        exp_vals[~finite_mask] = 0.0
        pair_val = v_min - np.log(exp_vals.sum()) / beta
        
        # Gradient of soft-min with respect to centers
        # d/dx_i [-1/beta * log(sum_j exp(-beta * d_ij / 2))]
        # = (1/beta) * sum_j [exp(-beta*d_ij/2) / sum_k exp(-beta*d_ik/2)] * (-beta/2) * d(d_ij)/dx_i
        # = -1/2 * sum_j [w_ij * (x_i - x_j) / d_ij]  where w_ij = exp(-beta*d_ij/2)/sum
        grad_pair = np.zeros_like(centers)
        for i in range(n):
            w_i = exp_vals[i, :] / max(exp_vals[i, :].sum(), 1e-15)
            for j in range(n):
                if i == j:
                    continue
                if not finite_mask[i, j]:
                    continue
                d_ij = dist[i, j]
                if d_ij < 1e-10:
                    continue
                grad_pair[i] += w_i[j] * (centers[i] - centers[j]) / d_ij * (-1.0)
        # Actually, let me redo this more carefully
        # soft_min = -1/beta * log(sum_j exp(-beta * d_ij / 2))
        # d(soft_min)/d(x_i) = -1/beta * 1/sum_j exp(-beta*d_ij/2) * sum_j exp(-beta*d_ij/2) * (-beta/2) * d(d_ij)/d(x_i)
        # = 1/2 * sum_j [exp(-beta*d_ij/2) / sum_k exp(-beta*d_ik/2)] * d(d_ij)/d(x_i)
        # d(d_ij)/d(x_i) = (x_i - x_j) / d_ij
        grad_pair = np.zeros_like(centers)
        for i in range(n):
            denom = exp_vals[i, :].sum()
            if denom < 1e-15:
                continue
            for j in range(n):
                if i == j:
                    continue
                if not finite_mask[i, j]:
                    continue
                d_ij = dist[i, j]
                if d_ij < 1e-10:
                    continue
                w = exp_vals[i, j] / denom
                grad_pair[i] += 0.5 * w * (centers[i] - centers[j]) / d_ij
    
    else:
        pair_val = 0.0
        grad_pair = np.zeros_like(centers)
    
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), axis=1)
    w_min = wall_dists.min()
    exp_wall = np.exp(-beta * (wall_dists - w_min))
    wall_val = w_min - np.log(exp_wall.sum()) / beta
    
    # Gradient of wall soft-min
    # wall_dist_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # This is a min of 4 terms per particle, which is tricky
    # Simplify: use the distance to nearest wall
    grad_wall = np.zeros_like(centers)
    for i in range(n):
        # Distance to each wall
        d_left = centers[i, 0] + 0.5
        d_right = 0.5 - centers[i, 0]
        d_bottom = centers[i, 1] + 0.5
        d_top = 0.5 - centers[i, 1]
        
        # Soft min of these 4 distances
        walls = np.array([d_left, d_right, d_bottom, d_top])
        wmin = walls.min()
        exp_w = np.exp(-beta * (walls - wmin))
        wsum = exp_w.sum()
        
        # Gradient contributions
        # d(d_left)/dx = 1, d(d_left)/dy = 0
        # d(d_right)/dx = -1, d(d_right)/dy = 0
        # d(d_bottom)/dx = 0, d(d_bottom)/dy = 1
        # d(d_top)/dx = 0, d(d_top)/dy = -1
        grad_wall[i, 0] += 0.5 * (exp_w[0] / wsum * 1.0 + exp_w[1] / wsum * (-1.0))
        grad_wall[i, 1] += 0.5 * (exp_w[2] / wsum * 1.0 + exp_w[3] / wsum * (-1.0))
    
    # Combine: the overall soft-min is min(pair_val, wall_val)
    # Approximate gradient as weighted combination
    total_val = min(pair_val, wall_val)
    if pair_val <= wall_val:
        grad = grad_pair.copy()
    else:
        grad = grad_wall.copy()
    
    return -total_val, -grad  # Return negative for minimization


def _grid_init():
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _random_init():
    """Random initialization inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.95


def _hex_init():
    """Hexagonal-ish lattice initialization."""
    pts = []
    n_per_row = 8
    for row in range(7):
        for col in range(n_per_row):
            x = (col + 0.5) / n_per_row - 0.5 + (0.5 / n_per_row if row % 2 else 0)
            y = (row + 0.5) / 7 - 0.5
            if abs(x) < 0.5 and abs(y) < 0.5:
                pts.append([x, y])
    pts = np.array(pts[:N])
    if len(pts) < N:
        extra = (np.random.rand(N - len(pts), 2) - 0.5) * 0.9
        pts = np.vstack([pts, extra])
    return pts[:N]


def _force_relax_vectorized(centers, max_iter=5000, dt=0.0005, deadline=None):
    """Vectorized force-directed relaxation."""
    centers = centers.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        inv_d3 = 1.0 / (dist ** 3 + 1e-10)
        np.fill_diagonal(inv_d3, 0.0)
        force = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        for k in range(2):
            wall_pos = centers[:, k] - 0.45
            wall_neg = centers[:, k] + 0.45
            force[:, k] -= 10.0 * np.maximum(wall_pos, 0) ** 2
            force[:, k] += 10.0 * np.maximum(wall_neg, 0) ** 2
        
        force -= 0.005 * centers
        
        centers += dt * force
        centers[:, 0] = np.clip(centers[:, 0], -0.495, 0.495)
        centers[:, 1] = np.clip(centers[:, 1], -0.495, 0.495)
    return centers


def _langevin_dynamics(centers, deadline, max_iter=100000, T0=0.02, T_end=0.001, dt=0.001):
    """Langevin dynamics with time-dependent temperature schedule."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _radius_fast(centers)
    T = T0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute forces from repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        # Repulsive force: 1/d^2 magnitude
        inv_d3 = 1.0 / (dist ** 3 + 1e-10)
        np.fill_diagonal(inv_d3, 0.0)
        force = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        for k in range(2):
            force[:, k] -= 20.0 * np.maximum(centers[:, k] - 0.45, 0) ** 2
            force[:, k] += 20.0 * np.maximum(-0.45 - centers[:, k], 0) ** 2
        
        # Langevin update: dx = force*dt + sqrt(2*T*dt)*noise
        noise = np.random.randn(*centers.shape) * np.sqrt(2.0 * T * dt)
        centers += force * dt + noise
        
        # Clamp to square
        centers[:, 0] = np.clip(centers[:, 0], -0.495, 0.495)
        centers[:, 1] = np.clip(centers[:, 1], -0.495, 0.495)
        
        # Cool down
        T = max(T_end, T0 * (T_end / T0) ** (it / max(max_iter, 1)))
        
        # Track best
        r = _radius_fast(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r


def _gradient_descent(centers, deadline, max_iter=5000, lr=0.001, beta=50.0):
    """Gradient descent on the soft-min radius."""
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _radius_fast(centers)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        neg_val, neg_grad = _soft_min_radius_and_grad(centers, beta)
        grad = -neg_grad  # We want to maximize, so follow negative gradient
        
        # Adaptive learning rate
        centers += lr * grad
        centers[:, 0] = np.clip(centers[:, 0], -0.495, 0.495)
        centers[:, 1] = np.clip(centers[:, 1], -0.495, 0.495)
        
        # Decrease learning rate
        lr *= 0.999
        
        r = _radius_fast(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r


def _local_refine(centers, deadline):
    """Local refinement using coordinate descent."""
    centers = centers.copy()
    best = centers.copy()
    best_r = _radius_fast(centers)
    
    for coord_iter in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for k in range(2):
                for step in [0.001, -0.001, 0.0005, -0.0005, 0.0002, -0.0002]:
                    new_c = best.copy()
                    new_c[i, k] += step
                    new_c[i, k] = np.clip(new_c[i, k], -0.495, 0.495)
                    r = _radius_fast(new_c)
                    if r > best_r:
                        best = new_c
                        best_r = r
                        improved = True
                        break
        if not improved:
            break
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 * DEADLINE_FRAC
    
    np.random.seed(42)
    
    # Multiple initialization strategies
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(),
        _random_init(),
    ]
    
    best_centers = None
    best_r = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Stage 1: Force relaxation
        relaxed = _force_relax_vectorized(init, max_iter=3000, dt=0.0003, deadline=deadline)
        
        # Stage 2: Langevin dynamics
        lang_centers, lang_r = _langevin_dynamics(relaxed, deadline, max_iter=50000, T0=0.015, T_end=0.0005, dt=0.0008)
        
        if lang_r > best_r:
            best_r = lang_r
            best_centers = lang_centers.copy()
        
        # Stage 3: Gradient descent on soft-min
        if time.time() < deadline - 10.0:
            gd_centers, gd_r = _gradient_descent(lang_centers, deadline, max_iter=3000, lr=0.002, beta=40.0)
            if gd_r > best_r:
                best_r = gd_r
                best_centers = gd_centers.copy()
        
        if time.time() > deadline:
            break
    
    # Final local refinement
    if best_centers is not None:
        refined, final_r = _local_refine(best_centers, deadline)
        if final_r > best_r:
            best_r = final_r
            best_centers = refined
    
    # Ensure validity
    if best_centers is None:
        best_centers = _grid_init()
    
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_centers
