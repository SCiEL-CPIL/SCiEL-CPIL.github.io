"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Generate Test Points (dense sampling of the sphere)
    # Using a higher resolution spiral for test points to better approximate the continuous sphere
    N_test = 5000
    i_test = np.arange(N_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / N_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_points = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Pre-compute tree for test points? No, we need distances from candidates to test points.
    # It is faster to build a tree on candidates and query test points, OR build tree on test points and query candidates?
    # We need, for each candidate, the set of test points for which it is the nearest.
    # This is equivalent to: For each test point, find the nearest candidate.
    
    best_points = points.copy()
    best_radius = float('inf')
    
    # Optimization Parameters
    max_iter = 2000
    step_size = 0.05
    decay = 0.995
    
    # Build initial KDTree on candidates
    tree = cKDTree(points)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Find nearest candidate for each test point
        # dists: shape (N_test, ), idx: shape (N_test, )
        dists, idx = tree.query(test_points)
        
        # Calculate covering radius (max distance from test point to nearest candidate)
        # Note: dists are Euclidean distances. Angular distance is arcsin(d/2) * 2? No, chord length d = 2*sin(theta/2).
        # theta = 2 * arcsin(d/2).
        # To minimize theta, we minimize d.
        current_radius = np.max(dists)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Calculate forces for each candidate
        # For each candidate i, we want to move it away from the test points that are assigned to it.
        # Specifically, we move it towards the "center of mass" of the voids? 
        # Standard Repulsion: Move point i in the direction of the vector sum of (test_point - point_i) 
        # for all test points assigned to i. This moves it towards the centroid of its Voronoi cell.
        # But we want to increase the minimum distance. 
        # Actually, if we move towards the centroid of the test points in its cell, we are moving into the "middle" of the region it covers.
        # This helps balance the distances.
        
        forces = np.zeros((N, 3))
        # Accumulate sum of (test_point - point_i) for points assigned to i
        # We can use bincount or scatter add.
        # Let's use a simple loop or numpy accumulation.
        # vectors = test_points - points[idx]  # Shape (N_test, 3)
        # This creates a large array. Let's try to be memory efficient.
        
        # Alternative: For each candidate, sum the vectors from itself to the test points assigned to it.
        # v_i = sum_{j in cell_i} (test_points[j] - points[i])
        # v_i = sum_{j in cell_i} test_points[j] - count_i * points[i]
        
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        
        # Efficient accumulation
        # np.add.at(sums, idx, test_points)
        # np.add.at(counts, idx, 1)
        
        np.add.at(sums, idx, test_points)
        np.add.at(counts, idx, 1)
        
        # Normalize counts to avoid division by zero
        safe_counts = np.where(counts > 0, counts, 1)
        
        # Centroid of the test points in each cell
        centroids = sums / safe_counts[:, np.newaxis]
        
        # Vector from point to its cell's test-point centroid
        # If a point has no test points (unlikely with 5000 test points and 50 candidates), force is 0
        vectors = centroids - points
        vectors[counts == 0] = 0.0
        
        # Normalize vectors to unit length to apply uniform step size direction?
        # Or use the magnitude?
        # If we move towards the centroid, we reduce the variance of distances within the cell.
        # The magnitude of the vector indicates how "off-center" the point is relative to its test points.
        # Let's normalize to unit vectors to prevent large jumps from distant cells, 
        # but scale by the local covering radius?
        
        # Simple approach: Move in direction of vectors with a fixed step size.
        # To avoid points collapsing, we can add a small repulsion from other candidates?
        # The "move to centroid" heuristic usually works well for covering problems.
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        unit_vectors = vectors / safe_norms
        
        # Update points
        points += step_size * unit_vectors
        
        # Re-normalize to unit sphere
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Update tree
        tree = cKDTree(points)
        
        # Decay step size
        step_size *= decay
        
    # Final check
    tree = cKDTree(best_points)
    dists, _ = tree.query(test_points)
    final_radius = np.max(dists)
    
    if final_radius < best_radius:
        best_points = points.copy()
        
    return best_points
# EVOLVE-BLOCK-END
