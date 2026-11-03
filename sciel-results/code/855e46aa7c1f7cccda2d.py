"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the max common radius for given centers in [-0.5, 0.5]^3."""
    n = len(centers)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Min pairwise distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(np.min(wall_dists), min_pair / 2.0)
    return r


def _initial_grid():
    """Create an initial grid placement."""
    m = 4  # 4x4x4 = 64 positions, use 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize(centers, max_time=100.0):
    """Optimize centers to maximize the minimum radius."""
    start_time = time.time()
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    # Local optimization using scipy
    def objective(x):
        """Negative radius to minimize."""
        c = x.reshape(-1, 3)
        # Quick check for validity
        if np.any(~np.isfinite(c)):
            return 1e10
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        if np.any(wall_dists < -1e-9):
            return 1e10
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        r = min(np.min(wall_dists), min_pair / 2.0)
        return -r
    
    x0 = current_centers.flatten()
    
    # Use Nelder-Mead for local optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        elapsed = time.time() - start_time
        if elapsed < max_time:
            new_centers = result.x.reshape(-1, 3)
            if np.all(np.isfinite(new_centers)):
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
    except Exception:
        pass
    
    # Try multiple random perturbations and local optimizations
    np.random.seed(42)
    for trial in range(50):
        elapsed = time.time() - start_time
        if elapsed > max_time - 5:
            break
        
        # Create perturbed version
        perturbation_scale = 0.01
        perturbed = best_centers + np.random.normal(0, perturbation_scale, (N, 3))
        # Clamp to valid range
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        try:
            result = minimize(objective, perturbed.flatten(), method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            elapsed = time.time() - start_time
            if elapsed < max_time:
                new_centers = result.x.reshape(-1, 3)
                if np.all(np.isfinite(new_centers)):
                    new_radius = _compute_radius(new_centers)
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centers = new_centers.copy()
        except Exception:
            continue
    
    return best_centers, best_radius


def construct():
    """Construct optimal packing of 50 spheres in unit cube."""
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    # Start with grid
    centers = _initial_grid()
    centers, radius = _optimize(centers, max_time=deadline - (time.time() - start_time))
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 2:
        return centers
    
    # Try a different initial configuration: body-centered arrangement
    np.random.seed(123)
    # Create clusters
    cluster_centers = np.array([
        [-0.25, -0.25, -0.25], [0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25], [0.25, -0.25, 0.25], [-0.25, 0.25, 0.25], [0.25, 0.25, 0.25]
    ])
    # Place 6-7 spheres per cluster
    cluster_configs = []
    for cc in cluster_centers:
        for dx, dy, dz in [
            [0, 0, 0], [0.08, 0, 0], [-0.08, 0, 0], [0, 0.08, 0], [0, -0.08, 0],
            [0, 0, 0.08], [0, 0, -0.08]
        ]:
            pos = cc + np.array([dx, dy, dz])
            if np.all(np.abs(pos) < 0.45):
                cluster_configs.append(pos)
            if len(cluster_configs) >= N:
                break
        if len(cluster_configs) >= N:
            break
    
    if len(cluster_configs) >= N:
        cluster_centers_arr = np.array(cluster_configs[:N])
        cluster_centers_arr, r2 = _optimize(cluster_centers_arr, max_time=deadline - (time.time() - start_time))
        if r2 > radius:
            centers = cluster_centers_arr
            radius = r2
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return centers
    
    # Ensure centers are within bounds
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
