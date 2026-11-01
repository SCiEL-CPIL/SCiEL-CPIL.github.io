"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

def _compute_covering_radius(points, probe_points):
    """
    Estimate the covering radius by finding the maximum distance 
    from any probe point to its nearest point in 'points'.
    """
    tree = cKDTree(points)
    dists, _ = tree.query(probe_points, k=1)
    return np.max(dists)

def _lloyd_relaxation(points, num_iters=50, probe_count=5000, seed=42):
    """
    Perform Lloyd's relaxation to minimize the covering radius.
    Uses a fixed set of probe points to approximate Voronoi centroids.
    """
    points = points.copy()
    
    # Generate probe points on the sphere. 
    # We use a Fibonacci sphere for even distribution of probes.
    i = np.arange(probe_count) + 0.5
    z = 1.0 - 2.0 * i / probe_count
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    probe_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for _ in range(num_iters):
        # Find nearest point for each probe
        tree = cKDTree(points)
        dists, indices = tree.query(probe_points, k=1)
        
        # Compute new centroids
        new_points = np.zeros_like(points)
        for i in range(N):
            mask = (indices == i)
            if np.sum(mask) > 0:
                # Centroid of the probe points assigned to this site
                centroid = np.mean(probe_points[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                else:
                    new_points[i] = points[i]
            else:
                # If no points assigned (should not happen with enough probes), keep old position
                new_points[i] = points[i]
        
        # Check convergence
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        if np.max(diff) < 1e-6:
            break
            
    return points

def _generate_initial_configurations():
    """
    Generate a set of initial configurations to optimize.
    Includes Fibonacci spiral and several random seeds.
    """
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random Configurations
    rng = np.random.RandomState(12345)
    for _ in range(5):
        # Generate random points on sphere
        pts = rng.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    return configs

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for evaluation overhead
    
    initial_configs = _generate_initial_configurations()
    
    best_points = None
    best_radius = float('inf')
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() - start_time > time_limit:
            break
            
        # Run Lloyd's relaxation
        # We use fewer iterations for later seeds if time is tight, but 50 is usually fast enough
        relaxed_pts = _lloyd_relaxation(init_pts, num_iters=50)
        
        # Evaluate covering radius
        # Use a dense probe set for accurate evaluation
        probe_count = 10000
        i = np.arange(probe_count) + 0.5
        z = 1.0 - 2.0 * i / probe_count
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        probe_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        radius = _compute_covering_radius(relaxed_pts, probe_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = relaxed_pts
            
    if best_points is None:
        # Fallback to first config if something went wrong
        best_points = initial_configs[0]
        
    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
