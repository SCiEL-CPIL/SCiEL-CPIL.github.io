import time
import numpy as np
from scipy.optimize import minimize
import random

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _repulsion_energy(x, eps=1e-6):
    """
    Calculate a smooth repulsion energy to be minimized.
    Low energy corresponds to points being far apart.
    """
    c = x.reshape(-1, 2)
    
    # Pairwise repulsion
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
    # Sum of inverse squared distances
    pair_energy = np.sum(1.0 / (dists_sq + eps))
    
    # Boundary repulsion
    # Distance to walls: 0.5 - |c_ik|
    # We want to maximize min distance to wall, so minimize 1/d^2
    dist_x = 0.5 - np.abs(c[:, 0])
    dist_y = 0.5 - np.abs(c[:, 1])
    
    # Clip to avoid singularities if a point is outside (shouldn't happen with bounds)
    dist_x = np.maximum(dist_x, 1e-6)
    dist_y = np.maximum(dist_y, 1e-6)
    
    wall_energy = np.sum(1.0 / (dist_x**2 + eps)) + np.sum(1.0 / (dist_y**2 + eps))
    
    return pair_energy + wall_energy

def _initial_jittered_grid(seed=42):
    """Generate a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    # Create grid points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N]
    
    # Add jitter
    jitter_scale = 0.5 / m * 0.8
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=grid.shape)
    grid = grid + jitter
    
    # Clip to bounds
    np.clip(grid, -0.5, 0.5, out=grid)
    
    return grid

def _optimize_repulsion(centers, deadline):
    """Minimize repulsion energy using L-BFGS-B."""
    x0 = centers.ravel()
    
    def obj(x):
        return _repulsion_energy(x)
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    try:
        # Use L-BFGS-B for bounded optimization
        # maxiter needs to be reasonable to fit in time
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < obj(x0):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centers

def _local_search_max_radius(centers, deadline, initial_step=0.05):
    """
    Local search to maximize the minimum radius.
    Uses a simulated annealing-like approach to escape local minima.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    rng = np.random.RandomState(12345)
    
    t_start = time.time()
    step_size = initial_step
    min_step = 1e-5
    
    # Number of iterations based on time
    # Each iteration involves O(N^2) distance calc. N=50 -> 2500 ops. Very fast.
    # We can do many iterations.
    
    while time.time() < deadline:
        # Pick a random point
        idx = rng.randint(N)
        orig = current_centers[idx].copy()
        
        # Generate a random perturbation
        # Bias the perturbation away from nearest neighbors? 
        # For simplicity, use Gaussian noise with adaptive step size.
        perturbation = rng.normal(0, step_size, size=2)
        new_pos = orig + perturbation
        
        # Keep within bounds
        np.clip(new_pos, -0.5, 0.5, out=new_pos)
        
        current_centers[idx] = new_pos
        new_radius = _compute_radius(current_centers)
        
        # Acceptance criterion
        if new_radius >= current_radius:
            # Always accept improvement or equal
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = current_centers.copy()
            # Shrink step size slightly on success
            step_size = max(min_step, step_size * 0.99)
        else:
            # Metropolis criterion: accept with probability exp((new - old) / T)
            # T should decay.
            # Since we are maximizing, delta = new - old is negative.
            # Let T be a "temperature" parameter.
            # We can fix T to a small value relative to radius scale.
            T = 0.001
            delta = new_radius - current_radius
            # If delta is very negative, prob is 0.
            if delta > -0.1: # Only consider if not a huge drop
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_radius = new_radius
                    # Do not update best here, as it's a worse move
                else:
                    # Revert
                    current_centers[idx] = orig
            else:
                current_centers[idx] = orig
            
            # Increase step size on failure (restarts)
            step_size = min(initial_step, step_size * 1.01)
            
        # Periodic restart if stuck? 
        # If we haven't improved best_radius in a long time, we could reset current to best.
        # For now, the Metropolis step handles this.

    return best_centers, best_radius

def construct():
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Initial Grid
    grid = _initial_jittered_grid(seed=SEED)
    r = _compute_radius(grid)
    best_centers = grid.copy()
    best_radius = r
    
    # 2. Repulsion Optimization on Grid
    if time.time() < deadline - 10:
        opt_grid = _optimize_repulsion(grid, deadline - 5)
        r = _compute_radius(opt_grid)
        if r > best_radius:
            best_radius = r
            best_centers = opt_grid.copy()
    
    # 3. Local Search on Best
    if time.time() < deadline - 2:
        final_centers, final_radius = _local_search_max_radius(best_centers, deadline - 1)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = final_centers
    
    # 4. Try a few random restarts if time permits
    remaining = deadline - time.time()
    if remaining > 10:
        n_attempts = min(5, int(remaining / 5)) # Max 5 attempts, 5s each
        for i in range(n_attempts):
            if time.time() > deadline - 1:
                break
            seed = SEED + 1000 + i
            rand_c = _initial_jittered_grid(seed=seed)
            
            # Quick repulsion opt
            opt_rand = _optimize_repulsion(rand_c, deadline - 1)
            r = _compute_radius(opt_rand)
            
            if r > best_radius:
                best_radius = r
                best_centers = opt_rand.copy()
                
                # Quick local search
                if time.time() < deadline - 2:
                    ls_c, ls_r = _local_search_max_radius(best_centers, deadline - 1)
                    if ls_r > best_radius:
                        best_radius = ls_r
                        best_centers = ls_c

    # Final Check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        best_centers = _initial_jittered_grid(seed=SEED)
        
    return best_centers
