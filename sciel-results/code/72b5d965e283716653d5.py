import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    
    return min(min_b, min_d)

def _pair_dist_matrix(p):
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    return np.linalg.norm(diffs, axis=2)

def _softmin_radius(p, beta=50.0):
    """Smooth approximation of min radius using softmin."""
    # Boundary
    b = 0.5 - np.max(np.abs(p), axis=1)
    
    # Pairwise
    dists = _pair_dist_matrix(p)
    np.fill_diagonal(dists, np.inf)
    pd = dists / 2.0
    
    # Softmin over all constraints
    all_constraints = np.concatenate([b, pd[np.triu_indices_from(pd, k=1)]])
    
    # Softmin: -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    softmin = min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return softmin

def _smooth_obj(p, beta=50.0):
    """Objective to maximize: negative softmin radius."""
    return -_softmin_radius(p, beta)

def _repulsion_anneal(p0, target_r, n_steps=200, time_limit=None, start_time=None):
    """Repulsion-based local search to push points apart while respecting target radius."""
    p = p0.copy()
    limits = 0.5 - target_r
    
    for step in range(n_steps):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        forces = np.zeros_like(p)
        
        # Boundary forces
        for axis in range(3):
            over_pos = p[:, axis] > limits
            over_neg = p[:, axis] < -limits
            forces[over_pos, axis] -= 1.0
            forces[over_neg, axis] += 1.0
        
        # Pairwise repulsion
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        mask = dists < 2 * target_r
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            unit_vecs = unit_vecs * mask[:, :, np.newaxis]
            pair_forces = np.sum(unit_vecs, axis=1)
            forces += pair_forces
        
        # Adaptive step size
        step_size = 0.02 * (1.0 - 0.5 * step / n_steps)
        p += step_size * forces
        
        # Clamp to cube
        p = np.clip(p, -0.5, 0.5)
    
    return p

def _seed_fcc():
    """Generate FCC lattice points and select the most compact cluster."""
    points = []
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    return points[idx[:N]].copy()

def _seed_grid():
    """Generate a 3D grid of points."""
    # 4x4x4 = 64 points, take 50
    coords = np.linspace(-0.4, 0.4, 4)
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
    grid -= np.mean(grid, axis=0)
    
    dists = np.linalg.norm(grid, axis=1)
    idx = np.argsort(dists)
    
    return grid[idx[:N]].copy()

def _seed_random(seed=42):
    """Generate random points."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(-0.4, 0.4, size=(N, 3))
    return p

def _seed_shell_core():
    """Generate points with a core-shell structure."""
    # Core: 27 points in a 3x3x3 grid
    core_coords = np.linspace(-0.25, 0.25, 3)
    core = np.array([[x, y, z] for x in core_coords for y in core_coords for z in core_coords])
    
    # Shell: remaining 23 points on a larger shell
    shell = []
    rng = np.random.RandomState(123)
    while len(shell) < N - len(core):
        p = rng.uniform(-0.4, 0.4, size=3)
        if np.linalg.norm(p) > 0.28:
            shell.append(p)
    
    p = np.vstack([core, np.array(shell)])
    p -= np.mean(p, axis=0)
    
    return p

def _local_search(p0, time_limit, start_time):
    """Perform local search to improve the configuration."""
    p = p0.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    # Try to increase radius in small steps
    dr = 0.0005
    
    while time.time() - start_time < time_limit:
        new_r = best_r + dr
        
        # Use repulsion annealing to try to achieve new_r
        p_trial = _repulsion_anneal(best_p, new_r, n_steps=150, 
                                      time_limit=time_limit - (time.time() - start_time),
                                      start_time=start_time)
        
        trial_r = _compute_radius(p_trial)
        
        if trial_r >= new_r - 1e-9:
            best_p = p_trial
            best_r = trial_r
        else:
            # Try smaller step
            dr = max(dr * 0.7, 0.0001)
            if dr < 0.0001:
                break
    
    return best_p, best_r

def construct():
    start_time = time.time()
    total_time_limit = 135.0  # Safety margin from 150s
    
    # Try multiple seeds and pick the best
    seeds = [
        ('fcc', _seed_fcc),
        ('grid', _seed_grid),
        ('random', _seed_random),
        ('shell_core', _seed_shell_core),
    ]
    
    best_p = None
    best_r = -1
    
    # Allocate time per seed
    time_per_seed = (total_time_limit - 5.0) / len(seeds)  # 5s for final polish
    
    for seed_name, seed_func in seeds:
        if time.time() - start_time > total_time_limit - 5.0:
            break
        
        p0 = seed_func()
        
        # Scale to fit in cube with some margin
        max_coord = np.max(np.abs(p0))
        if max_coord > 0:
            scale = 0.42 / max_coord
            p0 = p0 * scale
        
        # Local search
        seed_time_limit = min(time_per_seed, total_time_limit - 5.0 - (time.time() - start_time))
        if seed_time_limit < 5.0:
            break
        
        p_improved, r_improved = _local_search(p0, seed_time_limit, start_time)
        
        if r_improved > best_r:
            best_r = r_improved
            best_p = p_improved
    
    # Final polish with smooth optimization
    if best_p is not None and time.time() - start_time < total_time_limit - 3.0:
        # Try smooth optimization from best point
        p = best_p.copy()
        
        def obj(x):
            return _smooth_obj(x.reshape(N, 3), beta=30.0)
        
        def grad(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                g[i] = (obj(x_plus) - obj(x_minus)) / (2 * eps)
            return g
        
        # Run L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(obj, p.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 50, 'gtol': 1e-5})
            p_smooth = result.x.reshape(N, 3)
            r_smooth = _compute_radius(p_smooth)
            
            if r_smooth > best_r:
                best_r = r_smooth
                best_p = p_smooth
        except:
            pass
    
    # Final validation
    final_r = _compute_radius(best_p)
    if final_r > best_r:
        best_r = final_r
    
    return best_p
