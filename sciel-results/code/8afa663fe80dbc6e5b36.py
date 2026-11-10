"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def compute_radius(points):
    """Compute the maximum valid radius for given points."""
    if len(points) < 2:
        return 0.0
    # Minimum pairwise distance / 2
    from scipy.spatial.distance import pdist
    min_pair_dist = pdist(points, 'euclidean').min() / 2.0
    # Minimum distance to boundary
    min_boundary = (1.0 - np.linalg.norm(points, axis=1)).min()
    return min(min_pair_dist, min_boundary)

def objective(points_flat):
    """Objective to minimize (negative of radius)."""
    points = points_flat.reshape(-1, 2)
    # Add small penalty for points outside or too close to boundary
    dists = np.linalg.norm(points, axis=1)
    boundary_penalty = np.maximum(0, dists - 1.0)
    # Compute radius
    if len(points) < 2:
        return 1.0
    min_pair_dist = np.inf
    for i in range(len(points)):
        d = np.linalg.norm(points[i] - points, axis=1)
        d[i] = np.inf
        min_pair_dist = min(min_pair_dist, d.min())
    min_pair_dist /= 2.0
    min_boundary = (1.0 - dists).min()
    r = min(min_pair_dist, min_boundary)
    # Penalty for invalid configurations
    penalty = boundary_penalty.sum() * 100 + max(0, -min_boundary) * 100
    return -r + penalty

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Generate hexagonal lattice points
    def hex_lattice(n_target, max_radius=1.0):
        """Generate hexagonal lattice points within a circle."""
        points = []
        # Try different spacings
        for spacing in np.linspace(0.05, 0.15, 50):
            # Generate hexagonal lattice
            k = int(2.0 / spacing) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if np.hypot(x, y) <= max_radius - spacing / 2.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= n_target:
                # Sort by distance from center and take closest n_target
                dists = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists)[:n_target]
                return points[idx]
        return None
    
    # Initialize with hexagonal lattice
    points = hex_lattice(N)
    if points is None or len(points) < N:
        # Fallback: uniform random
        points = np.random.randn(N, 2) * 0.8
    
    # Optimize using local search
    best_points = points.copy()
    best_radius = compute_radius(points)
    
    # Use scipy minimize with L-BFGS-B
    x0 = points.flatten()
    
    # Define constraints and bounds
    bounds = [(-0.99, 0.99)] * (2 * N)
    
    # Try optimization with time check
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < objective(x0):
            new_points = result.x.reshape(-1, 2)
            new_radius = compute_radius(new_points)
            if new_radius > best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit - 10:
        return best_points
    
    # Try a few more optimization rounds with different starting points
    for restart in range(3):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Perturb current best
        perturbed = best_points + np.random.randn(N, 2) * 0.005
        # Clip to valid region
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.95
        perturbed[mask] = perturbed[mask] * (0.95 / norms[mask])
        
        x0 = perturbed.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < objective(x0):
                new_points = result.x.reshape(-1, 2)
                new_radius = compute_radius(new_points)
                if new_radius > best_radius:
                    best_points = new_points
                    best_radius = new_radius
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
