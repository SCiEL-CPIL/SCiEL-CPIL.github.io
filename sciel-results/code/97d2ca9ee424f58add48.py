import numpy as np
import time
from scipy.spatial import KDTree

N = 50

def _lloyd_relaxation(points, num_iterations=100, time_limit=100):
    """
    Perform Lloyd's relaxation on the sphere.
    points: (N, 3) array of points on the unit sphere.
    """
    start_time = time.time()
    points = points.copy()
    
    for _ in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Create a KDTree for fast nearest neighbor search
        tree = KDTree(points)
        
        # Generate sample points on the sphere to compute centroids
        # We can use a fine grid or random sampling
        # For efficiency, we'll use a moderate number of random points
        num_samples = 10000
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(num_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        # Assign each sample to its nearest point
        _, indices = tree.query(samples)
        
        # Compute centroids
        new_points = np.zeros_like(points)
        for i in range(N):
            mask = indices == i
            if np.sum(mask) > 0:
                centroid = np.sum(samples[mask], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[i] = centroid / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        # Check for convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        if diff < 1e-8:
            break
    
    return points

def construct():
    # Set a time limit to ensure we return before the 150s deadline
    total_time_limit = 140  # Leave some margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Run Lloyd's relaxation
    points = _lloyd_relaxation(points, num_iterations=50, time_limit=total_time_limit)
    
    # Normalize to ensure unit sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
