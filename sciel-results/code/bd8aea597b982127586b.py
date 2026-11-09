"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on unit sphere using Fibonacci spiral with optional angular offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _covering_radius_chord(points, test_points):
    """Compute covering radius as max chord distance (proportional to angular distance)."""
    tree = cKDTree(points)
    dists, _ = tree.query(test_points)
    return np.max(dists)

def _optimize(points, test_points, max_iter=500, initial_step=0.03, decay=0.997, deadline=None):
    """Optimize points to minimize covering radius using targeted force heuristic."""
    points = points.copy()
    n = len(points)
    
    best_points = points.copy()
    best_radius = _covering_radius_chord(points, test_points)
    
    tree = cKDTree(points)
    step_size = initial_step
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find nearest candidate for each test point
        dists, idx = tree.query(test_points)
        current_radius = np.max(dists)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Identify worst test points (top 10% by distance)
        n_worst = max(1, len(test_points) // 10)
        worst_indices = np.argsort(dists)[-n_worst:]
        
        # For each candidate, compute force based on:
        # 1. Weighted sum of vectors to worst test points in its cell
        # 2. Slight repulsion from other candidates to prevent collapse
        
        forces = np.zeros((n, 3))
        
        # Accumulate weighted vectors for worst points
        for j in worst_indices:
            i = idx[j]
            vec = test_points[j] - points[i]
            # Weight by how far this point is from the median distance in its cell
            forces[i] += vec
        
        # Add mild repulsion from other candidates to maintain spread
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-8:
                        forces[i] += 0.01 * d / dist
        
        # Normalize forces and apply
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_forces = forces / safe_norms
        
        # Adaptive step: scale by current radius relative to best
        radius_ratio = current_radius / (best_radius + 1e-10)
        effective_step = step_size * min(1.0, radius_ratio)
        
        points += effective_step * unit_forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        tree = cKDTree(points)
        step_size *= decay
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Test points: use a moderate density for speed
    N_test = 3000
    test_points = _fibonacci_sphere(N_test)
    
    best_points = None
    best_radius = float('inf')
    
    # Multiple restarts with different initializations
    restarts = [
        ("fibonacci", _fibonacci_sphere(N, offset=0.0)),
        ("fibonacci_rot", _fibonacci_sphere(N, offset=np.pi / 7)),
        ("random", _random_sphere(N, seed=42)),
    ]
    
    for name, init_points in restarts:
        if time.time() > deadline - 10:
            break
        
        # Coarse optimization
        opt_points = _optimize(init_points, test_points, max_iter=300, initial_step=0.04, 
                               decay=0.996, deadline=deadline)
        
        radius = _covering_radius_chord(opt_points, test_points)
        if radius < best_radius:
            best_radius = radius
            best_points = opt_points.copy()
    
    # Fine polish: use denser test points for the best configuration
    if best_points is not None and time.time() < deadline - 15:
        N_test_fine = 10000
        test_points_fine = _fibonacci_sphere(N_test_fine)
        
        polished = _optimize(best_points, test_points_fine, max_iter=200, initial_step=0.01,
                             decay=0.998, deadline=deadline)
        
        radius_fine = _covering_radius_chord(polished, test_points_fine)
        # Compare using same metric (chord distance)
        if radius_fine < best_radius:
            best_points = polished
    
    return best_points
# EVOLVE-BLOCK-END
