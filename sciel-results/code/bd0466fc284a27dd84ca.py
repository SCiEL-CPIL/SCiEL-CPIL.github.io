"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the packing score: min of pairwise distances/2 and distances to boundary."""
    n = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to boundary
    radii = np.sqrt(np.sum(points**2, axis=-1))
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n circles."""
    # Start with a reasonable spacing and expand until we have enough points
    for trial in range(50):
        spacing = 2.0 / (np.sqrt(n) + 1.0 - trial * 0.02)
        if spacing <= 0:
            break
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by distance from center to get a nice initial configuration
            idx = np.argsort(radius[mask])[:n]
            return points[idx]
    # Fallback: random points
    return np.random.randn(n, 2) * 0.5

def _random_seed(n, n_restarts=10):
    """Generate multiple random seeds."""
    seeds = []
    for _ in range(n_restarts):
        # Random points in the unit disk
        angles = np.random.uniform(0, 2*np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(points)
    return seeds

def _optimize(points, time_limit=100):
    """Optimize a starting configuration using gradient descent."""
    start_time = time.time()
    n = len(points)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # We want to maximize the minimum distance, so minimize negative
        # But we need to handle the constraint that points stay in the disk
        # Use a penalty method
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        radii = np.sqrt(np.sum(pts**2, axis=-1))
        # Penalty for going outside the disk
        penalty = np.sum(np.maximum(radii - 1.0, 0)**2)
        
        # Objective: maximize min distance, penalize boundary violations
        return -min_pair + 100.0 * penalty
    
    # Use L-BFGS-B for bounded optimization
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    result = minimize(objective, points.flatten(), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    # Project back to valid configuration
    pts = result.x.reshape(-1, 2)
    # Ensure points are strictly inside the disk
    radii = np.sqrt(np.sum(pts**2, axis=-1))
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] *= (1.0 - 1e-6) / radii[mask]
    
    return pts, result.fun

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5 seconds margin
    
    # Start with hexagonal seed
    best_points = _hexagonal_seed(N)
    best_score = _score(best_points)
    
    # Try to improve with optimization
    try:
        optimized, _ = _optimize(best_points, time_limit=30)
        opt_score = _score(optimized)
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    except Exception:
        pass
    
    # Try random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        seeds = _random_seed(N, n_restarts=5)
        for seed in seeds:
            if time.time() - start_time > deadline - 5:
                break
            try:
                optimized, _ = _optimize(seed, time_limit=15)
                opt_score = _score(optimized)
                if opt_score > best_score:
                    best_points = optimized
                    best_score = opt_score
            except Exception:
                pass
    
    # Final validation: ensure all points are inside the unit disk
    radii = np.sqrt(np.sum(best_points**2, axis=-1))
    if np.any(radii >= 1.0):
        best_points = best_points * (1.0 - 1e-8) / radii
    
    return best_points
# EVOLVE-BLOCK-END
