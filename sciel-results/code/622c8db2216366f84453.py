"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initial configurations (FCC-inspired, grid, random, shell-core)
2. Repulsion-based simulated annealing for global search
3. Voronoi Lloyd's algorithm for balanced spacing
4. Softmin gradient polishing for fine-tuning
5. Local search on binding constraints
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _seed_fcc():
    """
    Generate an FCC-inspired initial configuration for 50 spheres.
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
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_bcc():
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
                # BCC: corner + center
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

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    n_core = 20
    n_shell = N - n_core
    
    core = np.random.uniform(-0.25, 0.25, size=(n_core, 3))
    
    theta = np.random.uniform(0, 2 * np.pi, size=n_shell)
    phi = np.random.uniform(0, np.pi, size=n_shell)
    r = 0.35 + np.random.uniform(-0.05, 0.05, size=n_shell)
    shell = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    
    return np.vstack([core, shell])

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
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
    Returns (radius, gradient) where gradient is (N, 3).
    Uses log-sum-exp for numerical stability.
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Use log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)
    
    # Weights for pairs
    pair_log_weights = -pair_vals / t
    
    # Weights for walls
    wall_log_weights = -wall_vals / t
    
    # Combine for stability
    all_log_weights = np.concatenate([pair_log_weights, wall_log_weights.ravel()])
    max_lw = np.max(all_log_weights)
    if np.isinf(max_lw):
        return m_val, np.zeros_like(c)
    
    shifted = all_log_weights - max_lw
    log_sum = max_lw + np.log(np.sum(np.exp(shifted)))
    softmin_val = -t * log_sum
    
    # Gradient
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    pair_exp = np.exp(pair_log_weights - max_lw)
    wall_exp = np.exp(wall_log_weights - max_lw)
    
    total_exp = np.sum(pair_exp) + np.sum(wall_exp)
    
    if total_exp < 1e-300:
        return softmin_val, np.zeros_like(c)
    
    # Pair gradient
    pair_grad = -0.5 / t * pair_exp[:, np.newaxis] * unit_vecs
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] += pair_grad[idx]
        grad_c[j] -= pair_grad[idx]
    
    # Wall gradient
    signs = np.sign(c)
    grad_wall = -1.0 / t * wall_exp * signs
    grad_c += grad_wall
    
    grad_c /= total_exp
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=50):
    """
    Repulsion-based simulated annealing.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 15)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for it in range(iters_per_temp):
            if time.time() > deadline:
                break
            
            # Compute repulsive forces
            forces = np.zeros_like(c)
            
            # Pair repulsion
            diff = c[_iu] - c[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists * dists)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, np.newaxis] * unit_vecs
            
            for idx in range(_NUM_PAIRS):
                i, j = _PAIR_IDX[idx]
                forces[i] += pair_forces[idx]
                forces[j] -= pair_forces[idx]
            
            # Wall repulsion
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / (wall_dists * wall_dists)
            wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
            wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
            
            signs = np.sign(c)
            forces += wall_force_mag * signs
            
            # Add noise and update
            noise = np.random.normal(0, t * 0.1, size=c.shape)
            c_new = c + forces * 0.01 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            
            # Accept if better or with probability exp(-delta_r / t)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            delta_r = r_new - _compute_radius(c)
            if delta_r > 0 or np.random.random() < np.exp(-delta_r / t):
                c = c_new
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.001, n_temps=6):
    """
    Polish the configuration using gradient descent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
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
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
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

def _lloyd_step(c, iterations=100, clip_margin=0.001):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell.
    """
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
                center = np.array([0.0, 0.0, 0.0])
                direction = center - c[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                new_c[i] = c[i] + 0.01 * direction
                continue
            
            vertices = vor.vertices[region]
            
            if len(vertices) < 3:
                continue
            
            centroid = np.mean(vertices, axis=0)
            centroid = np.clip(centroid, -0.5 + clip_margin, 0.5 - clip_margin)
            new_c[i] = 0.7 * c[i] + 0.3 * centroid
        
        c = new_c
    
    return c

def _lloyd_optimize(c_init, deadline, max_rounds=50):
    """
    Run Lloyd's algorithm with periodic radius checks and restarts.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        c = _lloyd_step(c, iterations=20)
        
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        else:
            if round_idx % 5 == 4:
                c = best_c.copy()
                perturb_idx = np.random.choice(N, size=5, replace=False)
                for idx in perturb_idx:
                    c[idx] += np.random.normal(0, 0.02, size=3)
                c = np.clip(c, -0.49, 0.49)
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                c = best_c.copy()
                n_perturb = np.random.randint(2, 6)
                idxs = np.random.choice(N, size=n_perturb, replace=False)
                for idx in idxs:
                    c[idx] += np.random.normal(0, 0.015, size=3)
                c = np.clip(c, -0.499999, 0.499999)
                no_improve = 0
                continue
        
        # Find top-5 binding constraints
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        
        wall_dists = 0.5 - np.abs(c)
        
        n_top = min(5, len(dists))
        top_pair_indices = np.argsort(dists)[:n_top]
        
        wall_flat = wall_dists.ravel()
        top_wall_indices = np.argsort(wall_flat)[:min(5, len(wall_flat))]
        
        moved = False
        
        # Try moving apart top pair constraints
        for pi in top_pair_indices:
            i, j = _iu[pi], _ju[pi]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.003, 0.005, 0.008, 0.012):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    moved = True
                    break
            if moved:
                break
        
        # Try moving away from wall constraints
        if not moved:
            for wi in top_wall_indices:
                i, k = np.unravel_index(wi, wall_dists.shape)
                sign = 1.0 if c[i, k] > 0 else -1.0
                for step in (0.003, 0.005, 0.008, 0.012):
                    c_try = c.copy()
                    c_try[i, k] += step * sign
                    c_try = np.clip(c_try, -0.499999, 0.499999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
                        moved = True
                        break
                if moved:
                    break
        
        # If nothing moved, try a random small perturbation
        if not moved:
            idx = np.random.randint(N)
            c[idx] += np.random.normal(0, 0.008, size=3)
            c = np.clip(c, -0.499999, 0.499999)
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations - 5 diverse seeds
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_bcc(),
        _seed_shell_core()
    ]
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: 5 seeds, each gets roughly 1/5 of time minus final refinement
    n_seeds = len(seeds)
    final_reserve = 25.0  # seconds reserved for final refinement
    time_per_seed = (deadline - start_time - final_reserve) / n_seeds
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        # Stage 1: Repulsion annealing
        anneal_deadline = seed_start + 0.4 * time_per_seed
        c_anneal, r_anneal = _repulsion_anneal(s, anneal_deadline)
        
        # Stage 2: Lloyd's algorithm
        lloyd_deadline = seed_start + 0.7 * time_per_seed
        if time.time() < lloyd_deadline:
            c_lloyd, r_lloyd = _lloyd_optimize(c_anneal, lloyd_deadline, max_rounds=25)
        else:
            c_lloyd, r_lloyd = c_anneal, r_anneal
        
        # Stage 3: Polish
        polish_deadline = seed_start + 0.9 * time_per_seed
        if time.time() < polish_deadline:
            c_polish, r_polish = _polish(c_lloyd, polish_deadline, t_start=0.08, t_end=0.005, n_temps=5)
        else:
            c_polish, r_polish = c_lloyd, r_lloyd
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - final_reserve - 2.0:
            break
            
    # Final refinement on the global best
    final_start = time.time()
    final_deadline = deadline - 3.0
    
    # Final Lloyd's pass
    if time.time() < final_deadline - 15.0:
        c_lloyd_final, r_lloyd_final = _lloyd_optimize(best_c, final_deadline - 12.0, max_rounds=15)
        if r_lloyd_final > best_r:
            best_r = r_lloyd_final
            best_c = c_lloyd_final.copy()
    
    # Final high-precision polish
    if time.time() < final_deadline - 5.0:
        c_final, r_final = _polish(best_c, final_deadline - 3.0, t_start=0.03, t_end=0.002, n_temps=4)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # One last local search with many iterations
    if time.time() < final_deadline - 1.0:
        c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=800)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    return best_c
