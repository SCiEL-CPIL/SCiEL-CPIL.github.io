"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice)
2. Simulated annealing with repulsion to escape local minima
3. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
4. Smooth gradient-based polishing (soft-min via log-sum-exp)
5. Delaunay-based graph optimization for topological refinement
6. Final high-precision polish pass
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

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

def _get_active_constraints(c, r, tol=1e-4):
    """
    Identify indices of pairs and walls that are active (close to the minimum radius).
    Returns:
    pair_indices: list of (i, j) tuples
    wall_indices: list of (i, k) tuples
    """
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    # Active pairs: dist < 2*r + tol
    active_pair_mask = dists < (2.0 * r + tol)
    pair_indices = [(iu[0][k], iu[1][k]) for k in range(len(dists)) if active_pair_mask[k]]
    
    # Active walls: dist < r + tol
    wall_dists = 0.5 - np.abs(c)
    active_wall_mask = wall_dists < (r + tol)
    wall_indices = []
    for i in range(N):
        for k in range(3):
            if active_wall_mask[i, k]:
                wall_indices.append((i, k))
                
    return pair_indices, wall_indices

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Uses only unique pairs (i < j) to avoid double-counting.
    Returns (radius, gradient) where gradient is (N, 3).
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    # Use upper triangle for pairs
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq + 1e-16)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall_vals.ravel()
    
    # Concatenate all constraints
    all_vals = np.concatenate([pair_vals, wall_flat])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)
        
    # Stable log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)
    
    weights = exp_vals / s  # normalized weights
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    n_pairs = len(pair_vals)
    pair_w = weights[:n_pairs]  # weights for unique pairs
    
    pi, pj = iu
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # Vectorized: for each pair, compute contribution
    diff_pair = c[pi] - c[pj]  # shape (n_pairs, 3)
    
    # factor[k] = pair_w[k] / (2 * safe_dists[k])
    factor = pair_w / (2.0 * safe_dists)  # shape (n_pairs,)
    
    # Contribution to grad[pi[k]]: -factor[k] * diff_pair[k]
    # Contribution to grad[pj[k]]: +factor[k] * diff_pair[k]
    
    np.add.at(grad, pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, pj, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_w = weights[n_pairs:]  # N*3 weights
    wall_w = wall_w.reshape(N, 3)
    
    # d(wall_vals[i,k])/d(c[i,k]) = -sign(c[i,k])
    # grad[i,k] += wall_w[i,k] * sign(c[i,k])
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=5000):
    """
    Simulated annealing with repulsion to spread points apart.
    Uses adaptive perturbation size and tracks best radius.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.01
    
    temp = 0.05
    decay = 0.998
    pert_scale = 0.008
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute energy: sum of squared overlaps
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Target: all pair distances >= 2*r_target, wall distances >= r_target
        overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair ** 2) + np.sum(overlap_wall ** 2)
        
        # Adaptive perturbation: larger when stuck, smaller when improving
        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Compute new energy
        diff_new = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=-1))
        np.fill_diagonal(dists_new, np.inf)
        
        overlap_pair_new = np.maximum(0.0, 2.0 * r_target - dists_new)
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new ** 2) + np.sum(overlap_wall_new ** 2)
        
        # Accept
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                # Reduce perturbation on improvement
                pert_scale *= 0.95
            else:
                # Slightly increase perturbation if no improvement
                pert_scale = min(pert_scale * 1.02, 0.02)
        else:
            # No improvement, increase perturbation
            pert_scale = min(pert_scale * 1.03, 0.02)
        
        temp *= decay
        
        # Periodically try to increase r_target
        if it % 300 == 0 and E < 1e-4:
            r_target += 0.003
    
    return best_c, best_r

def _lloyds_algorithm(c_init, deadline, max_iters=200, step_size=0.1):
    """
    Lloyd's algorithm variant: move each center toward the centroid of its
    Voronoi cell (clipped to the cube) to equalize clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Add boundary points for Voronoi computation
    boundary_points = []
    eps = 0.1
    
    # Face centers extended outward
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign * (0.5 + eps)
            boundary_points.append(p)
    
    # Edge midpoints extended outward
    for i in range(3):
        for j in range(i+1, 3):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(3)
                    p[i] = si * (0.5 + eps)
                    p[j] = sj * (0.5 + eps)
                    boundary_points.append(p)
    
    # Corners extended outward
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = np.array([sx * (0.5 + eps), sy * (0.5 + eps), sz * (0.5 + eps)])
                boundary_points.append(p)
    
    boundary_points = np.array(boundary_points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        try:
            # Combine original points with boundary points for Voronoi
            all_points = np.vstack([c, boundary_points])
            vor = Voronoi(all_points)
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
            verts = np.clip(verts, -0.5 + 0.001, 0.5 - 0.001)
            
            # Compute centroid
            centroid = np.mean(verts, axis=0)
            
            # Move toward centroid with damping
            new_c[i] = (1 - step_size) * c[i] + step_size * centroid
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
    
    return best_c, best_r

def _delaunay_refinement(c_init, deadline, max_iters=100):
    """
    Delaunay-based graph optimization: use Delaunay edges to identify
    adjacent spheres and apply force-directed refinement to equalize
    the dual Voronoi cell volumes.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(c)
        except Exception:
            break
        
        # Build adjacency from Delaunay simplices
        adj = {i: set() for i in range(N)}
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    a, b = simplex[i], simplex[j]
                    adj[a].add(b)
                    adj[b].add(a)
        
        # For each edge, compute the midpoint and apply a spring force
        # that tries to make all edges have similar length (equal to 2r)
        r_current = _compute_radius(c)
        target_dist = 2.0 * r_current
        
        forces = np.zeros_like(c)
        
        for i in range(N):
            for j in adj[i]:
                if i < j:  # Only process each edge once
                    vec = c[i] - c[j]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-10:
                        continue
                    # Spring force: proportional to (dist - target_dist)
                    # If dist < target, push apart; if dist > target, pull together
                    # But we want to equalize, so use a soft spring
                    force_mag = 0.01 * (dist - target_dist) / max(dist, 1e-6)
                    unit_vec = vec / dist
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] < r_current:
                    sign = 1.0 if c[i, k] > 0 else -1.0
                    forces[i, k] += sign * 0.05 * (r_current - wall_dists[i, k])
        
        # Apply forces with damping
        c_new = c + forces
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
    
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

def _active_set_polish(c_init, deadline, max_iters=50):
    """
    Identify active constraints and perform local optimization on the involved points.
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
            
        pair_indices, wall_indices = _get_active_constraints(c, r, tol=1e-4)
        
        if not pair_indices and not wall_indices:
            break
            
        # Collect unique points involved
        active_points = set()
        for i, j in pair_indices:
            active_points.add(i)
            active_points.add(j)
        for i, k in wall_indices:
            active_points.add(i)
            
        if not active_points:
            break
            
        active_idx = sorted(list(active_points))
        n_active = len(active_idx)
        
        if n_active > 20: # Too many active points, skip or use a different strategy
            continue
            
        # Define objective for active points: maximize min distance among active constraints
        # We optimize the positions of active_points to maximize the minimum of the active constraints
        # This is a small-scale problem
        
        def active_obj(x_active):
            c_temp = c.copy()
            c_temp[active_idx] = x_active.reshape(n_active, 3)
            # Clip to ensure validity during optimization
            c_temp = np.clip(c_temp, -0.4999, 0.4999)
            
            # Compute radius based on ALL constraints, not just active ones,
            # to ensure we don't create new overlaps elsewhere
            return -_compute_radius(c_temp)
            
        def active_grad(x_active):
            c_temp = c.copy()
            c_temp[active_idx] = x_active.reshape(n_active, 3)
            c_temp = np.clip(c_temp, -0.4999, 0.4999)
            
            # Use soft-min gradient with very low temperature to approximate subgradient
            _, grad = _softmin_radius_and_grad(c_temp, 1e-4)
            return grad[active_idx].ravel()
            
        x0 = c[active_idx].flatten()
        bounds = [(-0.4999, 0.4999)] * (n_active * 3)
        
        try:
            res = minimize(
                active_obj,
                x0,
                jac=active_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            c_new = c.copy()
            c_new[active_idx] = res.x.reshape(n_active, 3)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
            else:
                # If no improvement, break to avoid cycling
                # Or try a different set of active points?
                # For simplicity, break if no progress in a few iterations
                pass
                
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
            for step in (0.002, 0.004, 0.008, 0.016):
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
            idx = np.argmin(dists)
            i, j = iu[0][idx], iu[1][idx]
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.004, 0.008, 0.016):
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
    Generate an FCC-like initial configuration.
    """
    # FCC lattice points in a cube
    a = 1.0 / 4.0  # lattice parameter
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                # Face centers
                if i < 3:
                    points.append((i*a - 0.5 + a/2, j*a - 0.5, k*a - 0.5))
                if j < 3:
                    points.append((i*a - 0.5, j*a - 0.5 + a/2, k*a - 0.5))
                if k < 3:
                    points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5 + a/2))
    
    points = np.array(points)
    # Filter to cube
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
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
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    # Use 3 seeds to allow more time for polishing
    time_per_seed = (deadline - start_time - 30.0) / len(seeds) # Reserve 30s for final polish
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - 35.0)
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=0.13)
        
        # Stage 2: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_anneal, seed_deadline)
        
        # Stage 3: Delaunay-based refinement
        c_delaunay, r_delaunay = _delaunay_refinement(c_lloyd, seed_deadline)
        
        # Stage 4: Smooth polish
        c_polish, r_polish = _smooth_polish(c_delaunay, seed_deadline, t_start=0.02, t_end=0.0005)
        
        # Stage 5: Local search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 35.0:
            break
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 10.0:
        final_deadline = deadline - 5.0
        # Active set polish is more targeted
        c_active, r_active = _active_set_polish(best_c, final_deadline, max_iters=20)
        if r_active > best_r:
            best_r = r_active
            best_c = c_active.copy()
            
        # Smooth polish with very low temperature
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.005, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # Delaunay refinement on final
        if time.time() < final_deadline - 5.0:
            c_del, r_del = _delaunay_refinement(best_c, final_deadline)
            if r_del > best_r:
                best_r = r_del
                best_c = c_del.copy()
            
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
