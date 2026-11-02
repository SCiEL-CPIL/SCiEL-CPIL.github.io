"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like, BCC, grid, random, shell-core)
2. Repulsion-based simulated annealing with vectorized forces
3. Force-based clearance equalization (replacing expensive Voronoi)
4. Smooth gradient-based polishing (soft-min via log-sum-exp) with analytical gradient
5. Epigraph formulation with auxiliary variable r
6. Final high-precision polish pass
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute upper triangle indices for pairwise distances
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
    Smooth approximation of the radius with analytical gradient (vectorized).
    
    Uses log-sum-exp to approximate min(values) where values are 
    pair distances/2 (each unique pair once) and wall distances.
    
    Returns (value, gradient) where gradient is wrt c (N, 3).
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # All constraint values
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    
    # Stable log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    value = m - t * np.log(s)
    
    grad = np.zeros_like(c)
    
    # Pair contributions (vectorized)
    weights_pair = exp_vals[:_NUM_PAIRS] / s  # shape (1225,)
    
    pair_diffs = c[_iu] - c[_ju]  # shape (1225, 3)
    pair_norms = np.sqrt(np.sum(pair_diffs ** 2, axis=1, keepdims=True))  # shape (1225, 1)
    pair_norms = np.maximum(pair_norms, 1e-12)
    
    # Unit vectors
    unit_vecs = pair_diffs / pair_norms  # shape (1225, 3)
    
    # Weighted sum: grad[i] += sum_j w_ij * unit_vec_ij, grad[j] -= sum_i w_ij * unit_vec_ij
    weighted_vecs = weights_pair[:, np.newaxis] * unit_vecs  # shape (1225, 3)
    
    # Scatter add
    np.add.at(grad, _iu, weighted_vecs)
    np.add.at(grad, _ju, -weighted_vecs)
    
    # Wall contributions
    weights_wall = exp_vals[_NUM_PAIRS:] / s  # shape (150,)
    wall_weights = weights_wall.reshape(N, 3)
    signs = np.sign(c)
    grad -= wall_weights * signs
    
    return value, grad

def _anneal_objective(x, r_target):
    """
    Objective for simulated annealing phase.
    Energy = sum over pairs max(0, r_target - d_ij)^2 + sum over walls max(0, r_target - w_i)^2
    """
    c = x.reshape(N, 3)
    energy = 0.0
    
    # Pairwise overlaps
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    # Overlap amount for each pair
    overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
    energy += np.sum(overlap_pair ** 2)
    
    # Wall overlaps
    wall_dists = 0.5 - np.abs(c)
    overlap_wall = np.maximum(0.0, r_target - wall_dists)
    energy += np.sum(overlap_wall ** 2)
    
    return energy

def _equalize_clearance(c, deadline, n_iter=30):
    """
    Force-based clearance equalization: move each sphere away from the
    nearest neighbor and walls proportionally to how tight the constraint is.
    Much cheaper than Voronoi-based Lloyd's algorithm.
    """
    c = c.copy()
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute clearance for each sphere (min distance to other spheres / 2, 
        # and distance to walls)
        clearances = np.zeros(N)
        for i in range(N):
            # Distance to all other spheres
            diffs = c - c[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            dists[i] = np.inf  # exclude self
            min_pair_dist = np.min(dists)
            
            # Distance to walls
            wall_d = 0.5 - np.abs(c[i])
            min_wall = np.min(wall_d)
            
            clearances[i] = min(min_pair_dist / 2.0, min_wall)
        
        # Find the most constrained sphere and push it
        min_clear_idx = np.argmin(clearances)
        
        # Compute repulsion forces for all points
        forces = _repulsion_forces(c)
        
        # Scale forces: points with lower clearance get pushed harder
        clearance_min = np.min(clearances)
        clearance_max = np.max(clearances)
        if clearance_max - clearance_min > 1e-10:
            # Normalized "tightness" - higher means more constrained
            tightness = 1.0 - (clearances - clearance_min) / (clearance_max - clearance_min)
            tightness = np.clip(tightness, 0.0, 1.0)
        else:
            tightness = np.ones(N)
        
        # Apply scaled forces
        step_size = 0.003 * (1.0 + 0.5 * it / max(n_iter, 1))
        c += forces * step_size * tightness[:, np.newaxis]
        c = np.clip(c, -0.49999, 0.49999)
    
    return c

def _repulsion_forces(c):
    """
    Compute repulsion forces for all points.
    Returns forces array of shape (N, 3).
    """
    forces = np.zeros_like(c)
    
    # Pair repulsion
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        force_mag = 1.0 / (dists * dists)
    force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs = np.where(np.isnan(unit_vecs), 0.0, unit_vecs)
    
    pair_forces = force_mag[:, np.newaxis] * unit_vecs
    
    np.add.at(forces, _iu, pair_forces)
    np.add.at(forces, _ju, -pair_forces)
    
    # Wall repulsion
    wall_dists = 0.5 - np.abs(c)
    with np.errstate(divide='ignore', invalid='ignore'):
        wall_force_mag = 1.0 / (wall_dists * wall_dists)
    wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
    wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
    
    signs = np.sign(c)
    forces += wall_force_mag * signs
    
    return forces

def _simulate_anneal(c_init, deadline, r_init=0.08, r_step=0.0003, max_iters=3000):
    """
    Perform simulated annealing to find a configuration that supports a growing radius.
    Uses repulsion-based energy with adaptive step size.
    Optimized: fewer expensive radius checks, cheaper equalization.
    """
    c = c_init.copy()
    current_r = r_init
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.05
    decay = 0.997
    step_scale = 0.02
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current energy (relatively cheap)
        E = _anneal_objective(c.flatten(), current_r)
        
        # Try to increase radius if feasible
        if E < 1e-4:
            current_r += r_step
            step_scale = max(0.004, step_scale * 0.995)
        else:
            temp *= decay
            step_scale = max(0.002, step_scale * 0.997)
        
        # Use repulsion forces for guidance
        forces = _repulsion_forces(c)
        
        # Perturbation: random Gaussian move + force-guided move
        move = np.random.normal(0.0, step_scale, size=c.shape)
        c_new = c + move + forces * 0.004
        c_new = np.clip(c_new, -0.499, 0.499)
        
        E_new = _anneal_objective(c_new.flatten(), current_r)
        
        # Accept if better or with probability exp(-dE/T)
        dE = E_new - E
        if dE < 0 or (temp > 1e-6 and np.random.random() < np.exp(-dE / temp)):
            c = c_new
            # Check radius less frequently (every 10th acceptance)
            if it % 10 == 0:
                r_check = _compute_radius(c)
                if r_check > best_r:
                    best_r = r_check
                    best_c = c.copy()
        
        # Occasionally do a cheap equalization step
        if it % 150 == 75:
            c_eq = _equalize_clearance(c, deadline, n_iter=5)
            r_eq = _compute_radius(c_eq)
            if r_eq > best_r:
                best_r = r_eq
                best_c = c_eq.copy()
                c = c_eq.copy()
                
    # Final radius check
    r_final = _compute_radius(c)
    if r_final > best_r:
        best_r = r_final
        best_c = c.copy()
        
    return best_c, best_r

def _epigraph_obj_and_grad(x, t):
    """
    Epigraph formulation: minimize -r subject to all constraint values >= r.
    We use a smooth penalty: maximize r - (1/t) * sum(softplus(r - v_j))^2
    where softplus(z) = log(1 + exp(z)).
    
    x = [c_0, c_1, ..., c_{N-1}, r]  (N*3 + 1 values)
    """
    c = x[:N*3].reshape(N, 3)
    r = x[N*3]
    
    # Compute all constraint values
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_dists = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    all_vals = np.concatenate([pair_dists, wall_vals.ravel()])
    
    # Softplus penalty: sum(softplus(r - v_j))^2
    z = r - all_vals
    # Stable softplus: softplus(z) = log(1+exp(z)) for z < 30, else z
    softplus_z = np.where(z < 30, np.log1p(np.exp(z)), z)
    penalty = np.sum(softplus_z ** 2)
    
    obj = -r + penalty / t
    
    # Gradient
    grad = np.zeros_like(x)
    
    # d(obj)/dr = -1 + (1/t) * 2 * sum(softplus_z * sigmoid(z))
    sigmoid_z = np.where(z < 30, np.exp(z) / (1 + np.exp(z)), 1.0)
    grad[N*3] = -1.0 + (2.0 / t) * np.sum(softplus_z * sigmoid_z)
    
    # d(obj)/dc: from penalty term
    # d(penalty)/d(c_k) = (2/t) * sum_j softplus(z_j) * sigmoid(z_j) * (-dv_j/dc_k)
    weights = (2.0 / t) * softplus_z * sigmoid_z  # shape (1375,)
    
    # Pair contributions
    weights_pair = weights[:_NUM_PAIRS]
    pair_diffs = c[_iu] - c[_ju]
    pair_norms = np.sqrt(np.sum(pair_diffs ** 2, axis=1, keepdims=True))
    pair_norms = np.maximum(pair_norms, 1e-12)
    unit_vecs = pair_diffs / pair_norms
    weighted_vecs = weights_pair[:, np.newaxis] * unit_vecs
    # d(penalty)/dc_i = -sum_j w_ij * unit_vec_ij (because dv/d(c_i) = unit_vec)
    np.add.at(grad[:N*3].reshape(N, 3), _iu, -weighted_vecs)
    np.add.at(grad[:N*3].reshape(N, 3), _ju, weighted_vecs)
    
    # Wall contributions: dv/d(c_ik) = -sign(c_ik)
    weights_wall = weights[_NUM_PAIRS:].reshape(N, 3)
    signs = np.sign(c)
    grad[:N*3] += (weights_wall * signs).reshape(N*3)
    
    return obj, grad

def _polish(c_init, deadline, t_start=0.05, t_end=0.002, n_temps=8):
    """
    Polish the configuration using two methods:
    1. Soft-min gradient descent
    2. Epigraph formulation with auxiliary variable r
    More aggressive: more iterations, tighter tolerances in final stages.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for idx, t in enumerate(temps):
        if time.time() > deadline:
            break
        
        # Later temperatures get more iterations and tighter tolerances
        progress = idx / max(n_temps - 1, 1)
        maxiter = int(150 + 150 * progress)
        ftol = 1e-14 if progress > 0.5 else 1e-13
        gtol = 1e-14 if progress > 0.5 else 1e-12
        
        # Method 1: Soft-min
        def obj_and_grad_softmin(x, t_val=t):
            c_reshaped = x.reshape(N, 3)
            val, grad = _softmin_radius_and_grad(c_reshaped, t_val)
            return -val, -grad
        
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj_and_grad_softmin,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.49999, 0.49999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            pass
        
        # Method 2: Epigraph
        if time.time() < deadline - 2.0:
            x0_epi = np.concatenate([c.flatten(), [best_r]])
            bounds_epi = [(-0.4999, 0.4999)] * (N * 3) + [(0.0, 0.5)]
            
            try:
                res = minimize(
                    _epigraph_obj_and_grad,
                    x0_epi,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds_epi,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
                )
                c_new = res.x[:N*3].reshape(N, 3)
                r_new = _compute_radius(c_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                    c = c_new.copy()
                    
            except Exception:
                pass
                
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    Also tries moving multiple points simultaneously and small random kicks.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Find the binding constraint
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        improved = False
        
        # If wall is binding, move the closest point away from the wall
        # If pair is binding, move the closest pair apart
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            # Direction away from wall
            sign = 1.0 if c[i, k] > 0 else -1.0
            # Try moving
            for step in (0.001, 0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.03):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        else:
            # Pair binding
            min_idx = np.argmin(dists)
            i, j = _iu[min_idx], _ju[min_idx]
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.001, 0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.03):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        
        # If no improvement, try perturbing several nearby points
        if not improved and it % 5 == 0:
            # Find the k closest pairs and perturb them
            k = min(3, len(dists))
            close_pairs = np.argsort(dists)[:k]
            c_try = c.copy()
            for pidx in close_pairs:
                i, j = _iu[pidx], _ju[pidx]
                vec = c_try[i] - c_try[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                    c_try[i] += 0.003 * vec
                    c_try[j] -= 0.003 * vec
                # Also add small random kick
                c_try[i] += np.random.normal(0, 0.002, size=3)
                c_try[j] += np.random.normal(0, 0.002, size=3)
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
    
    return best_c, best_r

def _initial_fcc_like(n_points):
    """
    Generate an FCC-like initial configuration with jitter.
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
    # Add jitter
    points += np.random.normal(0, 0.02, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    
    # Select n_points
    if len(points) > n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
        points = points[idx]
    else:
        # Pad with random points
        while len(points) < n_points:
            points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 3))])
    
    return points[:n_points]

def _initial_bcc(n_points):
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
    points += np.random.normal(0, 0.02, size=points.shape)
    points = np.clip(points, -0.45, 0.45)
    
    if len(points) > n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
        points = points[idx]
    else:
        while len(points) < n_points:
            points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 3))])
    
    return points[:n_points]

def _initial_grid(n_points):
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), n_points, replace=False)
    grid_pts = grid[idx] + np.random.normal(0, 0.01, size=(n_points, 3))
    return np.clip(grid_pts, -0.45, 0.45)

def _initial_random(n_points):
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, size=(n_points, 3))

def _initial_shell_core(n_points):
    """
    Generate a shell-core initial configuration.
    """
    n_core = 20
    n_shell = n_points - n_core
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin before 150s kill
    
    # Initial configurations - 5 diverse seeds
    seeds = []
    seeds.append(_initial_fcc_like(N))
    seeds.append(_initial_bcc(N))
    seeds.append(_initial_grid(N))
    seeds.append(_initial_random(N))
    seeds.append(_initial_shell_core(N))
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Allocate time: distribute across 5 seeds
    # First seed gets 30%, second 25%, third 20%, fourth 15%, fifth 10%
    time_alloc = [0.30, 0.25, 0.20, 0.15, 0.10]
    total_time = deadline - start_time
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 15.0:
            break
        
        seed_deadline = start_time + (sum(time_alloc[:i+1])) * total_time
        
        # Stage 1: Annealing (use 40% of seed time)
        anneal_deadline = start_time + (sum(time_alloc[:i]) + 0.40 * time_alloc[i]) * total_time
        c_anneal, r_anneal = _simulate_anneal(s, anneal_deadline, max_iters=3000)
        
        # Stage 2: Clearance equalization (use 15% of seed time)
        eq_deadline = start_time + (sum(time_alloc[:i]) + 0.55 * time_alloc[i]) * total_time
        c_eq = _equalize_clearance(c_anneal, eq_deadline, n_iter=20)
        r_eq = _compute_radius(c_eq)
        
        # Pick the better of anneal and equalized
        if r_eq > r_anneal:
            c_stage2 = c_eq
        else:
            c_stage2 = c_anneal
        
        if r_eq > best_r:
            best_r = r_eq
            best_c = c_eq.copy()
        
        # Stage 3: Polish with analytical gradient (use 30% of seed time)
        polish_deadline = start_time + (sum(time_alloc[:i]) + 0.85 * time_alloc[i]) * total_time
        c_polish, r_polish = _polish(c_stage2, polish_deadline, t_start=0.06, t_end=0.002, n_temps=6)
        
        # Stage 4: Local Search (remaining 15% of seed time)
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 15.0:
            break
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 15.0:
        final_deadline = deadline - 5.0
        remaining_time = final_deadline - time.time()
        
        # Use more temps if we have lots of time
        n_temps_final = 12 if remaining_time > 30 else 8
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.02, t_end=0.0005, n_temps=n_temps_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 5.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=800)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c
