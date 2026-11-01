"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Rigid-body transformations of high-symmetry seeds with per-point
breathing, smooth soft-min polishing via log-sum-exp, and fast local search.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using upper triangle only
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp formulation.
    Uses upper triangle for pairs to avoid double counting.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    # Pair distances (upper triangle)
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]  # shape (M, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Weights for pairs
    pair_weights = np.exp(-pair_vals / t)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        m = min(np.min(pair_vals), np.min(wall_vals))
        return m, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions: d/dx_i of dist(i,j)/2 = (x_i - x_j) / (2*dist)
    # But softmin uses -t*log(sum exp(-v/t)), so gradient of -softmin w.r.t. x_i
    # = sum_j w_ij * d(pair_val_ij)/dx_i / total_sum
    # d(pair_val_ij)/dx_i = (x_i - x_j) / (2 * dist_ij) * (-1) ... 
    # Actually: softmin = -t * log(sum exp(-v/t))
    # d(softmin)/dx = t * sum_j w_ij * d(v_ij)/dx / sum w
    # where w_ij = exp(-v_ij/t)
    # d(v_ij)/dx_i where v_ij = |x_i - x_j|/2 = (x_i - x_j)/(2*|x_i-x_j|)
    # So d(softmin)/dx_i = t * sum_j w_ij * (x_i - x_j)/(2*dist_ij) / total_sum
    # We want to MAXIMIZE softmin, so gradient ascent direction is +grad(softmin)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]  # (x_i - x_j)/dist
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # grad_pair[i] += w_ij * (x_i - x_j)/(2*dist) for each j
    pair_grad = np.zeros_like(c)
    np.add.at(pair_grad, iu[0], 0.5 * pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(pair_grad, iu[1], -0.5 * pair_weights[:, np.newaxis] * unit_vecs)
    
    grad_c += t * pair_grad
    
    # Wall contributions: v_ik = 0.5 - |x_ik|
    # d(v_ik)/dx_ik = -sign(x_ik)
    # d(softmin)/dx_ik = t * w_ik * (-sign(x_ik)) / total_sum
    signs = np.sign(c)
    wall_grad = -t * wall_weights * signs
    
    grad_c += wall_grad
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, r_target=0.13):
    """
    Repulsion-based simulated annealing to spread points apart.
    Uses vectorized operations for speed.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    r_curr = best_r
    
    temp = 0.05
    decay = 0.997
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute repulsion forces (vectorized)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: stronger when closer
        with np.errstate(divide='ignore', invalid='ignore'):
            force_scale = np.where(dists < 2*r_target, 1.0/(dists**2 + 1e-10), 0.0)
        force_scale[np.isnan(force_scale)] = 0.0
        
        forces = np.sum(force_scale[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion (vectorized)
        wall_dists = 0.5 - np.abs(c)
        wall_excess = np.maximum(0.0, r_target - wall_dists)
        wall_force = np.sign(c) * wall_excess * 2.0
        
        total_force = forces + wall_force
        
        # Random perturbation
        noise = np.random.normal(0.0, temp, size=c.shape)
        
        c_new = c + 0.01 * total_force + noise
        c_new = np.clip(c_new, -0.499, 0.499)
        
        # Evaluate
        r_new = _compute_radius(c_new)
        
        # Accept if better or with probability
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / temp):
            c = c_new
            r_curr = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= decay
        if temp < 0.001:
            temp = 0.001
    
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0005):
    """
    Polish the configuration using gradient ascent on the softmin radius
    with decreasing temperature for increasingly sharp approximation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Geometric progression of temperatures
    n_temps = 10
    temps = np.geomspace(t_start, t_end, n_temps)
    
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
            
        x0 = c.flatten()
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _lloyd_step(c, iterations=30, clip_margin=0.001):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell,
    clipped to the cube boundary. This equalizes clearance around each sphere.
    """
    from scipy.spatial import Voronoi
    
    c = c.copy()
    for _ in range(iterations):
        try:
            vor = Voronoi(c)
        except Exception:
            break
        
        new_c = c.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region - nudge toward center
                new_c[i] = c[i] * 0.9
                continue
            
            # Get vertices of the Voronoi cell
            verts = vor.vertices[region]
            
            if len(verts) == 0:
                new_c[i] = c[i] * 0.9
                continue
            
            # Clip vertices to cube
            verts = np.clip(verts, -0.5 + clip_margin, 0.5 - clip_margin)
            
            # Compute centroid
            centroid = np.mean(verts, axis=0)
            
            # Move toward centroid with damping
            new_c[i] = 0.7 * c[i] + 0.3 * centroid
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        c = new_c
    
    return c

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    Uses vectorized distance computation for speed.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Find the binding constraint (vectorized)
        iu = np.triu_indices(N, k=1)
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        # If wall is binding, move the closest point away from the wall
        # If pair is binding, move the closest pair apart
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            # Direction away from wall
            sign = 1.0 if c[i, k] > 0 else -1.0
            # Try moving
            for step in (0.002, 0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.4999, 0.4999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            # Pair binding
            min_idx = np.argmin(dists)
            i = iu[0][min_idx]
            j = iu[1][min_idx]
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.4999, 0.4999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    return best_c, best_r

def _seed_fcc():
    """
    Generate an FCC-like initial configuration with better spacing.
    """
    # FCC lattice: use a=0.25 for 4x4x4 grid
    a = 0.25
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x, y, z = i*a - 0.5, j*a - 0.5, k*a - 0.5
                # Corner
                points.append((x, y, z))
                # Face centers
                if i < 3:
                    points.append((x + a/2, y, z))
                if j < 3:
                    points.append((x, y + a/2, z))
                if k < 3:
                    points.append((x, y, z + a/2))
    
    points = np.array(points)
    # Filter to cube
    mask = np.all(np.abs(points) <= 0.48, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.015, size=(N, 3))
    else:
        # Pad with random points
        pad = np.random.uniform(-0.35, 0.35, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx] + np.random.normal(0, 0.01, size=(N, 3))

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, size=(N, 3))

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    c = np.zeros((N, 3))
    # Core: 8 points in inner region
    core = np.random.uniform(-0.15, 0.15, size=(8, 3))
    c[:8] = core
    # Shell: 42 points in outer region
    shell = np.random.uniform(-0.45, 0.45, size=(42, 3))
    c[8:] = shell
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    # Initial configurations - focus on diverse, structured seeds
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_random())
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Allocate time: more time for better seeds
    # Give first seed (FCC) more time as it's most promising
    time_budgets = [0.35, 0.25, 0.20, 0.20]
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 20.0:
            break
            
        seed_deadline = start_time + time_budgets[i] * (deadline - start_time)
        seed_deadline = min(seed_deadline, deadline - 20.0)
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=0.13)
        
        # Stage 2: Smooth polish
        c_polish, r_polish = _smooth_polish(c_anneal, seed_deadline, t_start=0.03, t_end=0.001)
        
        # Stage 3: Local search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=400)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 15.0:
        final_deadline = deadline - 8.0
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.005, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # Final local search
        if time.time() < final_deadline - 5.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
