"""Optimized sphere packing: 100 equal spheres in unit cube."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check if centers are inside the cube [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    tree = cKDTree(centers)
    # Get nearest neighbors for each point
    distances, _ = tree.query(centers, k=2)
    # distances[:, 1] is the distance to the nearest other point
    min_pair = np.min(distances[:, 1])
    return min(min_pair / 2.0, min_wall)

def _make_grid(n, offset=0.0):
    """Create a grid of n points in the unit cube."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    # Ensure we have at least n points
    while m**3 < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m - 0.5 + offset
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _local_optimization(centers, max_iters=5000, tol=1e-10):
    """Local optimization using gradient-free method with momentum."""
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Use a simple iterative repulsion-based method
    step_size = 0.01
    for it in range(max_iters):
        tree = cKDTree(centers)
        distances, indices = tree.query(centers, k=2)
        
        # Compute forces
        forces = np.zeros_like(centers)
        for i in range(N):
            j = indices[i, 1]
            if i == j:
                continue
            d = distances[i, 1]
            if d < 1e-10:
                # Random perturbation for coincident points
                forces[i] += np.random.randn(3) * 0.001
                continue
            # Repulsion force (stronger when close)
            force_mag = 1.0 / (d + 1e-6)
            direction = (centers[i] - centers[j]) / d
            forces[i] += force_mag * direction
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if centers[i, k] > 0.45:
                    forces[i, k] -= 0.5 * (centers[i, k] - 0.45)
                elif centers[i, k] < -0.45:
                    forces[i, k] += 0.5 * (-0.45 - centers[i, k])
        
        # Update positions
        centers = centers + step_size * forces
        # Keep within bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # Compute radius
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        # Decay step size
        step_size *= 0.999
        if step_size < 1e-8:
            break
    
    return best_centers, best_radius

def _anneal(centers, max_iters=10000, start_temp=0.01, end_temp=1e-6):
    """Simulated annealing for global optimization."""
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    temp = start_temp
    for it in range(max_iters):
        # Random perturbation
        new_centers = centers + np.random.randn(N, 3) * temp
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
        
        # Accept if better or with probability exp(-(new_radius - old_radius)/temp)
        if new_radius >= _compute_radius(centers) or np.random.random() < np.exp(-(new_radius - _compute_radius(centers)) / (temp + 1e-10)):
            centers = new_centers
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with a good initial configuration
    # Use a slightly perturbed grid
    np.random.seed(42)
    centers = _make_grid(N)
    # Add small random perturbations to break symmetry
    centers = centers + np.random.randn(N, 3) * 0.001
    centers = np.clip(centers, -0.5, 0.5)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Phase 1: Local optimization
    while time.time() < deadline - 20:
        centers, radius = _local_optimization(centers, max_iters=2000, tol=1e-10)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        else:
            # Try a different perturbation
            centers = best_centers + np.random.randn(N, 3) * 0.01
            centers = np.clip(centers, -0.5, 0.5)
    
    # Phase 2: Final polish with smaller steps
    while time.time() < deadline - 5:
        centers, radius = _local_optimization(best_centers, max_iters=5000, tol=1e-12)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        else:
            break
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
