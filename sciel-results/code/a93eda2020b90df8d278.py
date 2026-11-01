"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice, BCC, grid, random)
2. Vectorized repulsion annealing for global search
3. Delaunay-based refinement for targeted repulsion
4. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
5. Smooth gradient-based polishing (soft-min via log-sum-exp)
6. Epigraph formulation for direct t maximization
7. Final high-precision polish pass
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 50

# EVOLVE-BLOCK-START

# Precompute pair indices for efficiency
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Uses only unique pairs (i < j) to avoid double-counting.
    Returns (radius, gradient) where gradient is (N, 3).
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
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
    
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # Vectorized: for each pair, compute contribution
    diff_pair = c[_iu] - c[_ju]  # shape (n_pairs, 3)
    
    # factor[k] = pair_w[k] / (2 * safe_dists[k])
    factor = pair_w / (2.0 * safe_dists)  # shape (n_pairs,)
    
    # Contribution to grad[pi[k]]: -factor[k] * diff_pair[k]
    # Contribution to grad[pj[k]]: +factor[k] * diff_pair[k]
    
    np.add.at(grad, _iu, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, _ju, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_w = weights[n_pairs:]  # N*3 weights
    wall_w = wall_w.reshape(N, 3)
    
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _epigraph_objective(x, t):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    x = [c.flatten(), t] where c is (N,3) and t is scalar.
    Returns negative of t (to minimize) with penalty for constraint violations.
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Compute all constraint values
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    pair_vals = dists / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Violations: how much each constraint is below t
    violations = np.maximum(0, t - all_vals)
    
    # Penalize violations quadratically + maximize t
    penalty = np.sum(violations ** 2)
    
    return -t + penalty

def _epigraph_grad(x, t):
    """
    Gradient of epigraph objective.
    """
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    pair_vals = dists / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # violations = max(0, t - val)
    # d/dval: -2*violations if t > val else 0
    violations = np.maximum(0, t_val - all_vals)
    
    n_pairs = len(pair_vals)
    pair_viol = violations[:n_pairs]
    wall_viol = violations[n_pairs:]
    
    grad = np.zeros_like(x)
    
    # Pair contributions
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # d(pair_val)/d(c_i) = (c_i - c_j) / (2*dist)
    # d(obj)/d(c_i) += -2 * pair_viol[k] * (c_i - c_j) / (2*dist)
    #                = -pair_viol[k] * (c_i - c_j) / dist
    
    diff_pair = c[_iu] - c[_ju]
    factor = pair_viol / safe_dists  # shape (n_pairs,)
    
    np.add.at(grad[:N*3].reshape(N, 3), _iu, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad[:N*3].reshape(N, 3), _ju, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_viol_reshaped = wall_viol.reshape(N, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    # d(wall_val)/d(c[i,k]) = -sign(c[i,k])
    # d(obj)/d(c[i,k]) += -2 * wall_viol[i,k] * (-sign(c[i,k])) = 2 * wall_viol[i,k] * sign(c[i,k])
    grad[:N*3] += (2.0 * wall_viol_reshaped * signs).reshape(N*3)
    
    # d(obj)/dt = -1 + sum(2 * violations)
    grad[N*3] = -1.0 + 2.0 * np.sum(violations)
    
    return grad

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=5000):
    """
    Vectorized simulated annealing with repulsion to spread points apart.
    Uses adaptive perturbation size and tracks best radius.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.01
    
    temp = 0.05
    decay = 0.999
    pert_scale = 0.008
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute energy using only unique pairs (faster)
        diff = c[_iu] - c[_ju]  # (n_pairs, 3)
        dists_sq = np.sum(diff * diff, axis=1)  # (n_pairs,)
        
        # Target: all pair distances >= 2*r_target, wall distances >= r_target
        target_dist_sq = (2.0 * r_target) ** 2
        deficit_sq = np.maximum(0.0, target_dist_sq - dists_sq)
        overlap_pair = deficit_sq
        
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair) + np.sum(overlap_wall ** 2)
        
        # Adaptive perturbation
        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Compute new energy
        diff_new = c_new[_iu] - c_new[_ju]
        dists_sq_new = np.sum(diff_new * diff_new, axis=1)
        deficit_sq_new = np.maximum(0.0, target_dist_sq - dists_sq_new)
        overlap_pair_new = deficit_sq_new
        
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new) + np.sum(overlap_wall_new ** 2)
        
        # Accept
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            if it % 10 == 0:  # Check radius less frequently
                r_check = _compute_radius(c)
                if r_check > best_r:
                    best_r = r_check
                    best_c = c.copy()
                    pert_scale *= 0.95
            else:
                pert_scale = min(pert_scale * 1.01, 0.02)
        else:
            pert_scale = min(pert_scale * 1.02, 0.02)
        
        temp *= decay
        
        # Periodically try to increase r_target
        if it % 200 == 0 and E < 1e-3:
            r_target += 0.002
    
    # Final check
    r_final = _compute_radius(c)
    if r_final > best_r:
        best_r = r_final
        best_c = c.copy()
    
    return best_c, best_r

def _delaunay_refinement(c_init, deadline, max_iters=150):
    """
    Delaunay-based refinement: use Delaunay edges to identify close pairs
    and apply targeted repulsion along those edges to equalize distances.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(c)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a,b), max(a,b)))
        except Exception:
            break
        
        if len(edges) == 0:
            break
        
        # Apply targeted repulsion along Delaunay edges
        forces = np.zeros_like(c)
        
        for (i, j) in edges:
            vec = c[i] - c[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            vec = vec / dist
            # Repulsion strength inversely proportional to distance
            strength = 0.001 / max(dist, 0.01)
            forces[i] += strength * vec
            forces[j] -= strength * vec
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] < 0.05:
                    sign = 1.0 if c[i, k] > 0 else -1.0
                    forces[i, k] += sign * 0.001 / max(wall_dists[i, k], 0.01)
        
        # Apply forces with damping
        step = 0.1 / (1.0 + it * 0.01)
        c_new = c + step * forces
        c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            # If no improvement, try smaller step
            c_try = c + step * 0.5 * forces
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
    
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
        
        # Compute Voronoi diagram
        all_points = np.vstack([c, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except Exception:
            break
        
        # For each original point, find its Voronoi cell and compute centroid
        new_c = c.copy()
        
        for i in range(N):
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            
            # Get vertices of this region
            verts = vor.vertices[region]
            
            # Clip vertices to the cube [-0.5, 0.5]^3
            verts_clipped = np.clip(verts, -0.5, 0.5)
            
            # Compute centroid of the clipped region
            centroid = np.mean(verts_clipped, axis=0)
            
            # Move toward centroid
            new_c[i] = c[i] + step_size * (centroid - c[i])
            new_c[i] = np.clip(new_c[i], -0.499, 0.499)
        
        # Check if this improved the radius
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
        else:
            # If it got worse, revert
            c = best_c.copy()
        
        # Decrease step size
        step_size *= 0.95
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=50):
    """
    Polish the configuration using gradient descent on the softmin radius.
    Uses analytical gradient for speed.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Use more temperature steps for finer polishing
    n_temps = 25
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
            
        # More iterations for later (smaller t) temperatures
        max_iter = 80 + 80 * (ti // (n_temps // 3))
        
        def obj(x, t=t):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x, t=t):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
        
        x0 = c.flatten()
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _epigraph_polish(c_init, deadline):
    """
    Final high-precision polish using epigraph formulation.
    Maximizes t subject to all distance and wall constraints >= t.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Start with t slightly below current best radius
    t0 = best_r - 0.001
    x0 = np.concatenate([c.flatten(), [t0]])
    
    bounds = [(-0.499999, 0.499999)] * (N * 3) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            jac=_epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
        )
        x_new = res.x
        c_new = x_new[:N*3].reshape(N, 3)
        c_new = np.clip(c_new, -0.499999, 0.499999)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            
    except Exception:
        pass
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=300):
    """
    Local search: Perturb points that are constraining the radius.
    Enhanced: try multiple directions and steps, including random perturbations.
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
        diff = c[_iu] - c[_ju]  # (n_pairs, 3)
        dists = np.sqrt(np.sum(diff * diff, axis=1))  # (n_pairs,)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        min_pair_dist = np.min(dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            # Try moving away from wall
            for step in (0.001, 0.002, 0.005, 0.01):
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
            min_idx = np.argmin(dists)
            i, j = _iu[min_idx], _ju[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            # Try multiple steps and also random perturbations
            for step in (0.001, 0.002, 0.005, 0.01):
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
            else:
                # If no improvement, try random perturbation of the binding pair
                if it % 5 == 0:
                    c_try = c.copy()
                    rand_move = np.random.normal(0, 0.005, size=(2, 3))
                    c_try[i] += rand_move[0]
                    c_try[j] += rand_move[1]
                    c_try = np.clip(c_try, -0.499999, 0.499999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
    
    return best_c, best_r

def _initial_fcc_like():
    """
    Generate an FCC-like initial configuration for 50 points.
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
                for dx, dy, dz in [(0, 0, 0),
                                    (0.5, 0.5, 0),
                                    (0.5, 0, 0.5),
                                    (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    
    points = np.array(points)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_bcc():
    """
    Generate a BCC-inspired initial configuration.
    """
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
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _initial_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_initial_fcc_like())
    seeds.append(_initial_bcc())
    seeds.append(_initial_grid())
    seeds.append(_initial_random())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    total_time = deadline - start_time
    time_per_seed = total_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed * (1.0 + 0.1 * i), deadline - 25.0)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=best_r + 0.005, max_iters=4000)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 2: Delaunay refinement (new)
        c_delaunay, r_delaunay = _delaunay_refinement(c_anneal, seed_deadline, max_iters=100)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 3: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_delaunay, seed_deadline, max_iters=80)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 4: Polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.001)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 5: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 25.0:
            break
            
    # Final high-precision polish on the global best if time remains
    if time.time() < deadline - 25.0:
        final_deadline = deadline - 3.0
        
        # Epigraph polish first
        c_epi, r_epi = _epigraph_polish(best_c, min(final_deadline, time.time() + 30.0))
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
        
        if time.time() < final_deadline - 5.0:
            # Softmin polish
            c_final, r_final = _polish(best_c, final_deadline, t_start=0.01, t_end=0.00005)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 5.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        # Final ultra-fine polish
        if time.time() < final_deadline - 3.0:
            c_final2, r_final2 = _polish(best_c, final_deadline - 1.0, t_start=0.001, t_end=0.00001)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
        
        # Final epigraph pass
        if time.time() < final_deadline - 2.0:
            c_epi2, r_epi2 = _epigraph_polish(best_c, final_deadline - 0.5)
            if r_epi2 > best_r:
                best_r = r_epi2
                best_c = c_epi2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
