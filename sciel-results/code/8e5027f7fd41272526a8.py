import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize

N = 100
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 145.0
    return _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Soft-min objective: maximize the minimum of (pairwise_dist/2, wall_dist).
    We minimize the negative soft-min.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints into one array
    # Shape: (N, N + 3) roughly, but let's just flatten
    all_vals = np.concatenate([pair_dists.ravel(), wall_dists.ravel()])
    
    # Soft min via log-sum-exp trick:
    # min(x) ≈ - (1/beta) * log(sum(exp(-beta * x)))
    # To avoid overflow, subtract max first
    max_val = np.max(all_vals)
    soft_min_val = max_val - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - max_val))))
    
    # We want to maximize soft_min, so we minimize -soft_min
    return -soft_min_val

def _init_sobol():
    """
    Generate N points using a Sobol sequence (quasi-random, low discrepancy).
    """
    # Skip the first few points as they can be biased towards corners in some implementations
    sampler = Sobol(d=3, scramble=True, seed=42)
    # Generate slightly more than needed to skip initial points if desired, or just use N
    points = sampler.random(N)
    # Map [0, 1] to [-0.5, 0.5]
    centers = points * 1.0 - 0.5
    # Add tiny noise to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_bcc():
    """
    Build a BCC (body-centered cubic) lattice initialization.
    """
    nx, ny, nz = 5, 5, 2
    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = -0.5 + (i + 0.5) * (1.0 / nx)
                cy = -0.5 + (j + 0.5) * (1.0 / ny)
                cz = -0.5 + (k + 0.5) * (1.0 / nz)
                points.append([cx, cy, cz])
                points.append([cx + 0.5/nx, cy + 0.5/ny, cz + 0.5/nz])
    
    centers = np.array(points[:N])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _init_random(seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    return centers

def _polish_l_bfgs(centers, time_budget=10.0):
    """
    Local search using L-BFGS-B with soft-min objective.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    
    # Start with a moderate beta
    beta = 50.0
    
    while _time_left() > 1.0 and (time.time() - start_time) < time_budget:
        # Run L-BFGS for a fixed number of iterations or until time runs out
        # We'll run it in chunks to allow for beta adaptation
        
        # Define objective with current beta
        def obj(x):
            return _soft_min_objective(x, beta, None)
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Run minimize
        try:
            res = minimize(
                obj, 
                centers.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            centers = res.x.reshape(-1, 3)
            centers = np.clip(centers, -0.4999, 0.4999)
            
            r_current = _compute_radius(centers)
            
            if r_current > best_r + 1e-12:
                best_r = r_current
                best_centers = centers.copy()
                
            # Increase beta for sharper min approximation
            beta *= 1.5
            
            # If we made no progress, reduce beta? Or just break?
            # For now, just keep increasing beta to sharpen the landscape.
            
        except Exception:
            break
        
        # Check time
        if _time_left() < 2.0:
            break
            
    return best_centers, best_r

def _polish_force(centers, time_budget=10.0):
    """
    Force-directed local search (original approach).
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    start_time = time.time()
    
    # Parameters
    step_size = 0.01
    decay = 0.995
    min_step = 1e-6
    noise_scale = 0.001
    
    no_improve_count = 0
    max_no_improve = 100
    
    while _time_left() > 1.0 and (time.time() - start_time) < time_budget:
        r_current = _compute_radius(centers)
        
        if r_current > best_r + 1e-12:
            best_r = r_current
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Perturb to escape local minimum
            rng = np.random.default_rng()
            idx = rng.integers(0, N)
            centers[idx] = rng.uniform(-0.45, 0.45, 3)
            no_improve_count = 0
            continue

        forces = np.zeros_like(centers)
        
        # 1. Wall Forces
        wall_limit = max(r_current * 1.05, 0.01)
        for k in range(3):
            dists = 0.5 - np.abs(centers[:, k])
            mask = dists < wall_limit
            if np.any(mask):
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                strengths = (wall_limit - dists[mask]) / wall_limit
                forces[mask, k] += dirs * strengths * 2.0
        
        # 2. Pairwise Forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        limit = 2.0 * r_current * 1.05
        mask = (dists < limit) & (dists > 1e-9)
        
        strengths = np.zeros_like(dists)
        strengths[mask] = (limit - dists[mask]) / limit
        
        pair_forces = np.sum(diff * strengths[:, :, np.newaxis], axis=1)
        
        total_forces = forces + pair_forces
        
        # Add noise
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_scale, centers.shape)
        
        norms = np.linalg.norm(total_forces + noise, axis=1, keepdims=True)
        norms[norms == 0] = 1
        update_vec = (total_forces + noise) / norms
        
        centers += step_size * update_vec
        centers = np.clip(centers, -0.4999, 0.4999)
        
        step_size *= decay
        if step_size < min_step:
            step_size = min_step
            if no_improve_count > 20:
                break

    return best_centers, best_r

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializers
    inits = [
        ("sobol", _init_sobol),
        ("bcc", _init_bcc),
        ("rand1", lambda: _init_random(1)),
        ("rand2", lambda: _init_random(2)),
        ("rand3", lambda: _init_random(3)),
    ]
    
    # Allocate time. We have ~145s.
    # Give each init a chance.
    
    time_per_init = 20.0
    
    for name, init_func in inits:
        if _time_left() < time_per_init + 5.0:
            break
            
        centers = init_func()
        r_init = _compute_radius(centers)
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = centers.copy()
            
        # Run polish: try both L-BFGS and Force-directed
        # Split budget between them
        budget = min(time_per_init, _time_left() - 5.0)
        
        if budget > 2.0:
            # First try L-BFGS with half the budget
            half_budget = budget / 2.0
            polished_centers, polished_r = _polish_l_bfgs(centers, time_budget=half_budget)
            if polished_r > best_radius:
                best_radius = polished_r
                best_centers = polished_centers.copy()
            
            # Then try Force-directed with remaining budget
            remaining_budget = budget - half_budget
            if remaining_budget > 2.0:
                polished_centers2, polished_r2 = _polish_force(best_centers, time_budget=remaining_budget)
                if polished_r2 > best_radius:
                    best_radius = polished_r2
                    best_centers = polished_centers2.copy()
                
    # Final polish with all remaining time
    if best_centers is not None and _time_left() > 5.0:
        final_budget = _time_left() - 2.0
        # Use L-BFGS for final high-precision polish
        polished_centers, polished_r = _polish_l_bfgs(best_centers, time_budget=final_budget)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished_centers.copy()
            
    if best_centers is None:
        best_centers = _init_sobol()
        
    return best_centers
