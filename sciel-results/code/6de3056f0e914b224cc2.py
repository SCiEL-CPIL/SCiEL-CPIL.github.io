"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initial configurations (FCC-inspired, grid, random)
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

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    m = np.min(all_vals)
    
    pair_weights = np.exp(-pair_vals / t)
    np.fill_diagonal(pair_weights, 0.0)
    
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
        
    softmin_val = -t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    grad_pair_contrib = -0.5 * pair_weights[:, :, np.newaxis] * unit_vecs
    grad_c += np.sum(grad_pair_contrib, axis=1)
    
    signs = np.sign(c)
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, cooling=0.98, max_iters=5000):
    """
    Simulated annealing with repulsion forces to find a good global configuration.
    Uses a potential energy that penalizes close approaches and wall violations.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def energy(x):
        """Repulsion energy: sum of (1/d)^4 for pairs + wall penalties"""
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1e10)
        # Strong repulsion at short range
        pair_energy = np.sum(1.0 / (dists ** 4))
        # Wall penalty
        wall_dists = 0.5 - np.abs(x)
        wall_penalty = np.sum(np.maximum(0, 0.05 - wall_dists) ** 2)
        return pair_energy + 100 * wall_penalty
    
    t = t_start
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Propose a move: perturb a random point
        i = np.random.randint(N)
        delta = np.random.normal(0, 0.05 * t, size=3)
        c_new = c.copy()
        c_new[i] += delta
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        # Accept or reject
        e_old = energy(c)
        e_new = energy(c_new)
        de = e_new - e_old
        if de < 0 or np.random.random() < np.exp(-de / t):
            c = c_new
        
        # Update best
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        t *= cooling
        if t < t_end:
            t = t_end
    
    return best_c, best_r

def _lloyd_step(c, step_size=0.5):
    """
    Perform one Lloyd's algorithm iteration: move each center toward the centroid
    of its Voronoi cell, clipped to the cube boundary.
    """
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    
    c_new = c.copy()
    
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            continue
            
        verts = vor.vertices[region]
        
        if len(verts) == 0:
            continue
            
        centroid = np.mean(verts, axis=0)
        centroid = np.clip(centroid, -0.499999, 0.499999)
        
        c_new[i] = c[i] + step_size * (centroid - c[i])
        c_new[i] = np.clip(c_new[i], -0.499999, 0.499999)
    
    return c_new

def _lloyd_optimize(c_init, deadline, max_iters=300, step_decay=0.995):
    """
    Run Lloyd's algorithm iterations with decreasing step size.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    step_size = 0.3
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        c_new = _lloyd_step(c, step_size)
        
        diff = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-6:
            c_new += np.random.normal(0, 1e-4, size=c_new.shape)
            c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            step_size *= 0.95
        
        step_size *= step_decay
        step_size = max(step_size, 0.001)
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.001):
    """
    Polish the configuration using gradient descent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 8)
    
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
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
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
            
        r = _compute_radius(c)
        if r <= best_r:
            c = best_c.copy()
            r = best_r
             
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.005, 0.01, 0.02):
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
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.005, 0.01, 0.02):
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

def _seed_fcc():
    """
    Create an FCC-inspired initial configuration.
    FCC has 4 atoms per unit cell. We'll create a 4x4x4 grid of FCC-like positions.
    """
    # FCC basis positions in a unit cell
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    # Create positions in a 4x4x4 supercell
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for b in basis:
                    pos = np.array([i, j, k]) + b
                    positions.append(pos)
    
    positions = np.array(positions)
    # Normalize to fit in [-0.5, 0.5]^3
    positions = positions - positions.mean(axis=0)
    positions = positions * (1.0 / np.max(np.abs(positions))) * 0.45
    
    # Select 50 random positions
    idx = np.random.choice(len(positions), N, replace=False)
    return positions[idx]

def _seed_grid():
    """
    Create a grid-based initial configuration.
    """
    # Try different grid configurations
    configs = [
        (5, 5, 2),  # 50 points
        (5, 2, 5),
        (2, 5, 5),
        (4, 4, 4),  # 64 points, select 50
    ]
    
    best = None
    best_r = -1
    
    for (nx, ny, nz) in configs:
        if nx * ny * nz < N:
            continue
        
        ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
        ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
        ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
        
        grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
        
        if len(grid) > N:
            idx = np.random.choice(len(grid), N, replace=False)
            grid = grid[idx]
        
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best = grid
    
    return best if best is not None else np.random.uniform(-0.4, 0.4, size=(N, 3))

def _seed_random():
    """Random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_shell_core():
    """
    Create a shell-core configuration: some points on a spherical shell,
    others in the core. This can help with wall constraints.
    """
    # 10 points on a spherical shell
    shell_pts = []
    for i in range(10):
        # Random point on sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = 0.4
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        shell_pts.append([x, y, z])
    
    # 40 points in the core
    core_pts = np.random.uniform(-0.3, 0.3, size=(40, 3))
    
    return np.vstack([shell_pts, core_pts])

def _optimize_single(c_init, deadline):
    """
    Run the full optimization pipeline on a single initial configuration.
    """
    # Stage 1: Repulsion annealing for global search
    c1, r1 = _repulsion_anneal(c_init, deadline, t_start=1.0, t_end=0.05, max_iters=2000)
    
    # Stage 2: Lloyd's algorithm for balanced spacing
    c2, r2 = _lloyd_optimize(c1, deadline, max_iters=200)
    
    # Stage 3: Polish with softmin gradient
    c3, r3 = _polish(c2, deadline, t_start=0.05, t_end=0.005)
    
    # Stage 4: Local search on binding constraints
    c4, r4 = _local_search(c3, deadline, max_iters=200)
    
    return c4, r4

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate diverse initial configurations
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_shell_core(),
        np.random.uniform(-0.45, 0.45, size=(N, 3)),
    ]
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    time_per_seed = (deadline - start_time) / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        c_opt, r_opt = _optimize_single(s, seed_deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
        if time.time() > deadline - 5.0:
            break
            
    # Final polishing on the global best if time remains
    if time.time() < deadline - 5.0:
        final_deadline = deadline - 2.0
        
        # Try Lloyd's again on the best configuration
        c_lloyd2, r_lloyd2 = _lloyd_optimize(best_c, final_deadline, max_iters=150)
        if r_lloyd2 > best_r:
            best_r = r_lloyd2
            best_c = c_lloyd2.copy()
        
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.03, t_end=0.003)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 2.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
