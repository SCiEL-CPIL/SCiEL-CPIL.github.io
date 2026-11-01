"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Hybrid of Global Repulsion Annealing, Geometric Lloyd's Algorithm (Voronoi),
and Sharp Gradient Polishing using a smooth approximation of the minimum distance
(soft-min) to directly target the bottleneck constraint.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# Precompute indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    This is a smooth approximation of the minimum distance.
    As t -> 0, softmin -> min.
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Combine pair and wall constraints into a single vector for softmin
    # We treat wall distances as "distances" too.
    # To keep it simple and vectorized, we compute weights separately.
    
    # Softmin value: -t * log( sum(exp(-v/t)) )
    # We need to handle the sum of pairs and walls.
    
    # To avoid overflow in exp(-v/t) when t is small, we can shift by min_val
    # But for t > 0.001 and v < 0.5, exp is safe.
    
    pair_weights = np.exp(-pair_vals / t)
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        # Fallback to hard min
        min_pair = np.min(pair_vals)
        min_wall = np.min(wall_vals)
        m = min(min_pair, min_wall)
        return m, np.zeros_like(c)
        
    softmin_val = -t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    # Gradient of softmin w.r.t. c
    # d/dc (-t log(sum exp(-v/t))) = sum (exp(-v/t) / sum) * (dv/dc)
    # Note: v is the "cost" (distance). We want to maximize v.
    # The gradient of the softmin value w.r.t. c is:
    # grad = (1/total_sum) * sum (w_i * grad(v_i))
    
    # Pair contributions
    # v_pair_ij = |c_i - c_j| / 2
    # grad_{c_i} v_pair_ij = 0.5 * (c_i - c_j) / |c_i - c_j|
    # grad_{c_j} v_pair_ij = 0.5 * (c_j - c_i) / |c_i - c_j|
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # w_ij = exp(-v_ij/t)
    # Contribution to grad_c[i] from pair (i,j): w_ij * 0.5 * unit_vec_ij
    # Contribution to grad_c[j] from pair (i,j): w_ij * 0.5 * (-unit_vec_ij)
    
    pair_grad_term = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_c, _iu, pair_grad_term)
    np.add.at(grad_c, _ju, -pair_grad_term)
    
    # Wall contributions
    # v_wall_ik = 0.5 - |c_ik|
    # grad_{c_ik} v_wall_ik = -sign(c_ik)
    # w_ik = exp(-v_ik/t)
    # Contribution to grad_c[i,k]: w_ik * (-sign(c_ik))
    
    signs = np.sign(c)
    wall_grad_term = -wall_weights * signs
    
    grad_c += wall_grad_term
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _seed_fcc():
    """
    Generate an FCC-like initial configuration.
    """
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_grid():
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_bcc():
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    return points

def _seed_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _repulsion_anneal(c_init, deadline, r_target=0.13):
    """
    Repulsion-based simulated annealing to spread points apart.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    r_curr = best_r
    
    temp = 0.05
    decay = 0.995
    max_iters = 3000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        np.fill_diagonal(dists, np.inf)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            force_scale = np.where(dists < 2*r_target, 1.0/(dists**2 + 1e-10), 0.0)
        force_scale[np.isnan(force_scale)] = 0.0
        
        forces = np.sum(force_scale[:, :, np.newaxis] * diff, axis=1)
        
        wall_dists = 0.5 - np.abs(c)
        wall_excess = np.maximum(0.0, r_target - wall_dists)
        wall_force = np.sign(c) * wall_excess * 2.0
        
        total_force = forces + wall_force
        
        noise = np.random.normal(0.0, temp, size=c.shape)
        
        c_new = c + 0.01 * total_force + noise
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / max(temp, 1e-6)):
            c = c_new
            r_curr = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= decay
        if temp < 0.001:
            temp = 0.001
            
    return best_c, best_r

def _lloyd_refine(c_init, deadline, max_iters=50, step_size=0.5):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell.
    This equalizes clearance and is a structural geometric refinement.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        try:
            vor = Voronoi(c)
        except Exception:
            break
        
        new_c = c.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region: nudge towards center or keep
                # To avoid sticking to walls, we can project towards center
                new_c[i] = c[i] * 0.95
                continue
            
            verts = vor.vertices[region]
            if len(verts) == 0:
                continue
            
            # Clip vertices to cube to ensure centroid is inside
            verts = np.clip(verts, -0.5, 0.5)
            centroid = np.mean(verts, axis=0)
            
            # Move towards centroid
            new_c[i] = c[i] + step_size * (centroid - c[i])
            
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
        else:
            # If no improvement, reduce step size or break
            # Simple strategy: if stuck, break
            if r_new < best_r - 1e-6:
                 break
                 
    return best_c, best_r

def _gradient_polish(c_init, deadline, t_start=0.05, t_end=0.001):
    """
    Polish using gradient ascent on the soft-min radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    n_temps = 8
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
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
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

def _local_search(c_init, deadline, max_iters=300):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.002, 0.005, 0.01):
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
            min_idx = np.argmin(dists)
            i = _iu[min_idx]
            j = _ju[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.005, 0.01):
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_bcc(),
        _seed_random()
    ]
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Time allocation
    # 4 seeds, ~35s each
    time_budget_per_seed = 35.0
    
    for s in seeds:
        if time.time() > deadline - 10.0:
            break
            
        seed_deadline = time.time() + time_budget_per_seed
        seed_deadline = min(seed_deadline, deadline - 10.0)
        
        # 1. Global Repulsion
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline - 15.0)
        
        # 2. Geometric Lloyd Refinement
        # Use remaining time for this seed, reserve 5s for polish
        lloyd_deadline = seed_deadline - 5.0
        c_lloyd, r_lloyd = _lloyd_refine(c_anneal, lloyd_deadline, max_iters=100)
        
        # 3. Gradient Polish
        c_polish, r_polish = _gradient_polish(c_lloyd, seed_deadline, t_start=0.02, t_end=0.0005)
        
        # 4. Final Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final global polish if time remains
    if time.time() < deadline - 10.0:
        final_deadline = deadline - 5.0
        c_final, r_final = _gradient_polish(best_c, final_deadline, t_start=0.005, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()

    return best_c

# EVOLVE-BLOCK-END
