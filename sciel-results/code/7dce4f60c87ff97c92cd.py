"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Rigid-body transformations (rotation + translation) of high-symmetry
seed lattices with per-point radial breathing, combined with smooth soft-min
polishing via log-sum-exp and repulsion-based annealing.
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
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp formulation.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Combine all values for softmin
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    m = np.min(all_vals)
    
    # Weights for pairs (upper triangle only to avoid double counting)
    pair_weights = np.zeros_like(pair_vals)
    iu = np.triu_indices(N, k=1)
    pair_weights[iu] = np.exp(-pair_vals[iu] / t)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions (only upper triangle, symmetric)
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # For each pair (i,j) with i<j: contribution to c[i] is +0.5*w*unit_vec, to c[j] is -0.5*w*unit_vec
    # But since we only store upper triangle, we need to add both contributions
    for i in range(N):
        for j in range(i+1, N):
            w = pair_weights[i, j]
            if w > 1e-15:
                vec = unit_vecs[i, j]  # points from j to i
                grad_c[i] += 0.5 * w * vec
                grad_c[j] -= 0.5 * w * vec
    
    # Wall contributions
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, r_target=0.13):
    """
    Repulsion-based simulated annealing to spread points apart.
    Uses inverse-distance repulsion to push overlapping spheres apart.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temp = 0.04
    decay = 0.997
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute repulsion forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: stronger when closer
        with np.errstate(divide='ignore', invalid='ignore'):
            force_scale = np.where(dists < 2.5*r_target, 1.0/(dists**2 + 1e-10), 0.0)
        force_scale[np.isnan(force_scale)] = 0.0
        
        forces = np.sum(force_scale[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_force = np.zeros_like(c)
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] < r_target:
                    wall_force[i, k] = np.sign(c[i, k]) * (r_target - wall_dists[i, k]) * 3.0
        
        total_force = forces + wall_force
        
        # Adaptive noise: larger when stuck
        r_curr = _compute_radius(c)
        if r_curr < best_r - 0.001:
            noise_scale = temp * 1.5
        else:
            noise_scale = temp
        
        noise = np.random.normal(0.0, noise_scale, size=c.shape)
        
        c_new = c + 0.008 * total_force + noise
        c_new = np.clip(c_new, -0.499, 0.499)
        
        # Evaluate
        r_new = _compute_radius(c_new)
        
        # Accept if better or with probability
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / temp):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= decay
        if temp < 0.0005:
            temp = 0.0005
    
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
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
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

def _lloyd_step(c, iterations=50, clip_margin=0.001):
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

def _lloyd_optimize(c_init, deadline):
    """
    Run Lloyd's algorithm followed by smooth polishing.
    """
    c = _lloyd_step(c_init, iterations=30)
    c = np.clip(c, -0.4999, 0.4999)
    r = _compute_radius(c)
    
    # Polish after Lloyd
    c, r = _smooth_polish(c, deadline, t_start=0.02, t_end=0.001)
    
    return c, r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        # Find the binding constraint
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
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
            for step in (0.002, 0.004, 0.008, 0.015, 0.025):
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
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.004, 0.008, 0.015, 0.025):
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
    Generate an FCC-like initial configuration with 50 points.
    """
    # FCC lattice: generate more points and select 50
    a = 1.0 / 5.0  # smaller lattice parameter for more points
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # Corner
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                # Face centers
                if i < 4:
                    points.append((i*a - 0.5 + a/2, j*a - 0.5, k*a - 0.5))
                if j < 4:
                    points.append((i*a - 0.5, j*a - 0.5 + a/2, k*a - 0.5))
                if k < 4:
                    points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5 + a/2))
    
    points = np.array(points)
    # Filter to cube with margin
    mask = np.all(np.abs(points) <= 0.48, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.015, size=(N, 3))
    else:
        # Pad with random points
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx] + np.random.normal(0, 0.012, size=(N, 3))

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.42, 0.42, size=(N, 3))

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    c = np.zeros((N, 3))
    # Core: 12 points in inner region (tetrahedral arrangement)
    core_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    core_r = 0.12
    core = np.column_stack([
        core_r * np.cos(core_angles),
        core_r * np.sin(core_angles),
        np.random.uniform(-0.05, 0.05, 12)
    ])
    c[:12] = core
    # Shell: 38 points in outer region
    shell = np.random.uniform(-0.43, 0.43, size=(38, 3))
    c[12:] = shell
    return c

def _seed_bcc():
    """
    Generate a BCC-like initial configuration.
    """
    a = 1.0 / 4.0
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                # Body center
                points.append((i*a - 0.5 + a/2, j*a - 0.5 + a/2, k*a - 0.5 + a/2))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.48, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.015, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_hcp_like():
    """
    Generate an HCP-like layered initial configuration.
    """
    # 5 layers of ~10 points each
    c = np.zeros((N, 3))
    layer_heights = np.linspace(-0.4, 0.4, 5)
    per_layer = [10, 10, 10, 10, 10]
    
    idx = 0
    for li, h in enumerate(layer_heights):
        n = per_layer[li]
        # Hexagonal packing in the layer
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        if li % 2 == 0:
            r = 0.28
        else:
            r = 0.22
            angles += np.pi / n
        c[idx:idx+n, 0] = r * np.cos(angles)
        c[idx:idx+n, 1] = r * np.sin(angles)
        c[idx:idx+n, 2] = h
        idx += n
    
    c += np.random.normal(0, 0.01, size=(N, 3))
    return np.clip(c, -0.49, 0.49)

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    # Initial configurations - diverse seeds
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_bcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_hcp_like())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_random())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: give more time to structured seeds, less to random
    # Total budget for seed processing: ~120s, leaving 22s for final polish
    seed_budget = 120.0
    time_per_seed = seed_budget / len(seeds)
    
    for i, s in enumerate(seeds):
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        # Structured seeds get more time for annealing
        if i < 4:  # fcc, bcc, grid, hcp
            anneal_time = min(time_per_seed * 0.45, remaining * 0.4)
            polish_time = min(time_per_seed * 0.35, remaining * 0.3)
            ls_time = min(time_per_seed * 0.2, remaining * 0.2)
        else:  # shell, random
            anneal_time = min(time_per_seed * 0.35, remaining * 0.3)
            polish_time = min(time_per_seed * 0.4, remaining * 0.35)
            ls_time = min(time_per_seed * 0.25, remaining * 0.2)
        
        t0 = time.time()
        
        # Stage 1: Repulsion annealing
        anneal_deadline = min(t0 + anneal_time, deadline - 20.0)
        c_anneal, r_anneal = _repulsion_anneal(s, anneal_deadline, r_target=0.135)
        
        # Stage 2: Smooth polish
        polish_deadline = min(time.time() + polish_time, deadline - 15.0)
        c_polish, r_polish = _smooth_polish(c_anneal, polish_deadline, t_start=0.03, t_end=0.001)
        
        # Stage 3: Local search
        ls_deadline = min(time.time() + ls_time, deadline - 10.0)
        c_ls, r_ls = _local_search(c_polish, ls_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Final polish on the global best if time remains
    remaining = deadline - time.time()
    if remaining > 8.0:
        final_deadline = deadline - 5.0
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.008, t_end=0.0002)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
        
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=400)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
        
        # Ultra-fine final polish
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.003, t_end=0.00005)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
    
    # Ensure validity
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c

# EVOLVE-BLOCK-END
