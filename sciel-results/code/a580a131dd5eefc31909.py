"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(points):
    """Compute the largest common radius for the given circle centers."""
    n = len(points)
    # Minimum distance between any pair of centers
    min_dist = np.inf
    for i in range(n):
        diffs = points[i+1:] - points[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if len(dists) > 0:
            min_dist = min(min_dist, np.min(dists))
    # Minimum distance from any center to the container boundary
    radii = np.sqrt(np.sum(points**2, axis=1))
    min_boundary = np.min(1.0 - radii)
    return min(min_dist / 2.0, min_boundary)


def _hexagonal_packing(n):
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98


def _anneal(points, max_time=100.0):
    """Simulated annealing to improve the packing."""
    start_time = time.time()
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        # Pick a random circle to move
        idx = np.random.randint(n)
        
        # Random perturbation
        dx = np.random.normal(0, T)
        dy = np.random.normal(0, T)
        
        new_points = points.copy()
        new_points[idx] += np.array([dx, dy])
        
        # Ensure the moved circle stays inside the container
        r = np.sqrt(new_points[idx, 0]**2 + new_points[idx, 1]**2)
        if r > 0.999:
            new_points[idx] *= 0.999 / r
        
        new_r = _compute_radius(new_points)
        delta_r = new_r - best_r
        
        if delta_r > 0:
            points = new_points
            best_points = new_points.copy()
            best_r = new_r
        else:
            # Accept with probability exp(delta_r / T)
            if np.random.rand() < np.exp(delta_r / T):
                points = new_points
        
        T *= cooling
    
    return best_points, best_r


def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    start_time = time.time()
    
    # Generate initial hexagonal packing
    points = _hexagonal_packing(N)
    
    # Run simulated annealing
    best_points, best_r = _anneal(points, max_time=100.0)
    
    # Check time and return best result
    if time.time() - start_time > 110.0:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
