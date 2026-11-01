"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice)
2. Simulated annealing with repulsion to escape local minima
3. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
4. Smooth gradient-based polishing (soft-min via log-sum-exp) with analytical gradient
5. Final high-precision polish pass
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

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
    Smooth approximation of the radius with analytical gradient.
    
    Uses log-sum-exp to approximate min(values) where values are 
    pair distances/2 (each unique pair once) and wall distances.
    
    Returns (value, gradient) where gradient is wrt c (N, 3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Only upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0  # shape (1225,)
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3) = 150 values
    
    # All constraint values: 1225 pairs + 150 walls = 1375
    all_vals = np.concatenate([pair_dists, wall_vals.ravel()])
    
    # Stable log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    value = m - t * np.log(s)
    
    # Gradient: d/dx [m - t*log(s)] where m = min(all_vals), s = sum(exp(-(v-m)/t))
    # d(value)/d(c_k) = -t * (1/s) * sum_j [d(exp(-(v_j-m)/t))/d(c_k)]
    # d(exp(-(v_j-m)/t))/d(c_k) = exp(-(v_j-m)/t) * (-1/t) * (dv_j/d(c_k) - dm/d(c_k))
    # For simplicity, since m is piecewise constant (subgradient), we use:
    # grad = -t * (1/s) * sum_j exp(-(v_j-m)/t) * (-1/t) * dv_j/d(c_k)
    #       = (1/s) * sum_j exp(-(v_j-m)/t) * dv_j/d(c_k)
    
    grad = np.zeros_like(c)
    
    # Pair contributions: for pair (i,j), v_ij = |c_i - c_j| / 2
    # dv_ij/d(c_i) = (c_i - c_j) / |c_i - c_j|
    # dv_ij/d(c_j) = (c_j - c_i) / |c_i - c_j|
    weights_pair = exp_vals[:len(pair_dists)] / s  # shape (1225,)
    
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        w = weights_pair[idx]
        grad[i] += w * d / dist
        grad[j] -= w * d / dist
    
    # Wall contributions: for sphere i, coord k, v = 0.5 - |c_ik|
    # dv/d(c_ik) = -sign(c_ik)
    weights_wall = exp_vals[len(pair_dists):] / s  # shape (150,)
    wall_weights = weights_wall.reshape(N, 3)
    signs = np.sign(c)
    grad -= wall_weights * signs
    
    return value, grad

def _anneal_objective(x, r_target):
    """
    Objective for simulated annealing phase.
    We want to minimize the "overlap" energy if the target radius is not satisfied.
    Energy = sum over pairs max(0, r_target - d_ij)^2 + sum over walls max(0, r_target - w_i)^2
    If we can achieve r_target with zero energy, we can increase r_target.
    """
    c = x.reshape(N, 3)
    energy = 0.0
    
    # Pairwise overlaps
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Overlap amount for each pair
    # If dist < 2*r_target, overlap is (2*r_target - dist)
    # Squared overlap to make it smooth
    overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
    energy += np.sum(overlap_pair ** 2)
    
    # Wall overlaps
    wall_dists = 0.5 - np.abs(c)
    overlap_wall = np.maximum(0.0, r_target - wall_dists)
    energy += np.sum(overlap_wall ** 2)
    
    return energy

def _lloyd_step(c, deadline, n_iter=50):
    """
    Lloyd's algorithm: move each sphere to the centroid of its Voronoi cell,
    clipped to stay within bounds. This equalizes clearance.
    """
    c = c.copy()
    for it in range(n_iter):
        if time.time() > deadline:
            break
        try:
            vor = Voronoi(c)
            new_c = c.copy()
            for i in range(N):
                region = vor.regions[vor.point_region[i]]
                if -1 in region or len(region) == 0:
                    continue
                vertices = vor.vertices[region]
                if len(vertices) == 0:
                    continue
                # Centroid of Voronoi cell
                centroid = np.mean(vertices, axis=0)
                # Clip to stay in cube with margin
                new_c[i] = np.clip(centroid, -0.45, 0.45)
            c = new_c
        except Exception:
            break
    return c

def _simulate_anneal(c_init, deadline, r_init=0.08, r_step=0.0003, max_iters=3000):
    """
    Perform simulated annealing to find a configuration that supports a growing radius.
    Uses repulsion-based energy with adaptive step size.
    """
    c = c_init.copy()
    current_r = r_init
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.05
    decay = 0.998
    step_scale = 0.02
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current energy
        E = _anneal_objective(c.flatten(), current_r)
        
        # Try to increase radius if feasible
        if E < 1e-4:
            current_r += r_step
            step_scale = max(0.005, step_scale * 0.99)
        else:
            temp *= decay
            step_scale = max(0.003, step_scale * 0.995)
        
        # Perturbation: random Gaussian move
        move = np.random.normal(0.0, step_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.499, 0.499)
        
        E_new = _anneal_objective(c_new.flatten(), current_r)
        
        # Accept if better or with probability exp(-dE/T)
        dE = E_new - E
        if dE < 0 or (temp > 1e-6 and np.random.random() < np.exp(-dE / temp)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
        
        # Occasionally do a Lloyd step to equalize
        if it % 200 == 100:
            c_lloyd = _lloyd_step(c, deadline, n_iter=3)
            r_lloyd = _compute_radius(c_lloyd)
            if r_lloyd > best_r:
                best_r = r_lloyd
                best_c = c_lloyd.copy()
                c = c_lloyd.copy()
                
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.002, n_temps=8):
    """
    Polish the configuration using gradient descent on the softmin radius
    with analytical gradient.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        def obj_and_grad(x):
            c_reshaped = x.reshape(N, 3)
            val, grad = _softmin_radius_and_grad(c_reshaped, t)
            return -val, -grad
        
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.49999, 0.49999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

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
            for step in (0.003, 0.006, 0.01, 0.02):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
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
            
            for step in (0.003, 0.006, 0.01, 0.02):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    return best_c, best_r

def _initial_fcc_like(n_points):
    """
    Generate an FCC-like initial configuration with jitter.
    FCC lattice points in a unit cell, then jitter and select.
    """
    # FCC lattice: corners + face centers
    # In [-0.5, 0.5]^3, FCC points per unit cell: 8 corners * 1/8 + 6 faces * 1/2 = 1 + 3 = 4
    # We'll generate a 3x3x3 grid of FCC unit cells
    pts = []
    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                base = np.array([ix, iy, iz]) * 0.5
                # Corners
                for dx in [0, 0.25]:
                    for dy in [0, 0.25]:
                        for dz in [0, 0.25]:
                            p = base + np.array([dx, dy, dz])
                            if -0.49 <= p[0] <= 0.49 and -0.49 <= p[1] <= 0.49 and -0.49 <= p[2] <= 0.49:
                                pts.append(p)
                # Face centers
                for face in range(3):
                    for val in [0.125, 0.375]:
                        p = base + np.zeros(3)
                        p[face] = val
                        if -0.49 <= p[0] <= 0.49 and -0.49 <= p[1] <= 0.49 and -0.49 <= p[2] <= 0.49:
                            pts.append(p)
    
    pts = np.array(pts)
    # Add jitter
    pts += np.random.normal(0, 0.02, size=pts.shape)
    pts = np.clip(pts, -0.45, 0.45)
    
    # Select n_points
    if len(pts) > n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    else:
        # Pad with random points
        while len(pts) < n_points:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, size=(1, 3))])
    
    return pts[:n_points]

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # 12s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. FCC-like structured
    seeds.append(_initial_fcc_like(N))
    
    # 2. Jittered simple cubic
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    grid_pts = grid[idx] + np.random.normal(0, 0.01, size=(N, 3))
    seeds.append(np.clip(grid_pts, -0.45, 0.45))
    
    # 3. Random
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    time_per_seed = (deadline - start_time) / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        # Stage 1: Annealing
        c_anneal, r_anneal = _simulate_anneal(s, seed_deadline, max_iters=2500)
        
        # Stage 2: Lloyd's algorithm
        c_lloyd = _lloyd_step(c_anneal, seed_deadline, n_iter=20)
        r_lloyd = _compute_radius(c_lloyd)
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c_lloyd.copy()
        
        # Stage 3: Polish with analytical gradient
        c_polish, r_polish = _polish(c_lloyd if r_lloyd > r_anneal else c_anneal, seed_deadline, t_start=0.08, t_end=0.003, n_temps=6)
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 3.0:
            break
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 8.0:
        final_deadline = deadline - 3.0
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.03, t_end=0.001, n_temps=8)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=800)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
