"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Iterative Minimax Refinement
    # We use a large number of random test points to approximate the Voronoi cells
    # and identify the worst-case directions.
    
    n_test = 20000
    iterations = 50
    alpha = 0.5  # Step size
    
    for _ in range(iterations):
        # Generate random test points on the sphere
        # Using uniform distribution on sphere
        u = np.random.uniform(size=(n_test, 3))
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        
        # Compute distances from test points to data points
        # cos_d = u @ points.T
        # min_idx = np.argmax(cos_d, axis=1)
        # min_cos = np.max(cos_d, axis=1)
        # min_ang = np.arccos(np.clip(min_cos, -1, 1))
        
        # Efficiently:
        cos_d = u @ points.T
        min_idx = np.argmax(cos_d, axis=1)
        min_cos = np.max(cos_d, axis=1)
        
        # For each data point, we need to find the test point assigned to it that is farthest.
        # This is equivalent to finding the min cosine among the assigned points.
        
        # Initialize arrays to store the farthest point for each data point
        farthest_points = np.zeros_like(points)
        farthest_cos = np.ones(N)  # Max possible cosine is 1 (angle 0)
        
        # We can use np.minimum.at or similar, but let's just loop or use vectorized ops
        # Since N=130 is small, we can do this efficiently.
        
        # For each test point j assigned to i = min_idx[j]:
        # If cos_d[j, i] < farthest_cos[i], update farthest_cos[i] and farthest_points[i]
        
        # Vectorized approach using scatter/min
        # We want the min of cos_d[j, min_idx[j]] for each i.
        
        # Create a mask or use bincount? 
        # Simpler: loop over test points is too slow in Python.
        # Let's use a trick:
        # Create an array of indices [0..n_test-1]
        # We can use np.argsort on the distances? No.
        
        # Let's use a loop over the N data points? No, that's O(N * n_test) which is 130 * 20000 = 2.6M ops. This is fine in NumPy if vectorized, but hard to vectorize the "min per group" without bincount-like ops which don't support "min value from another array".
        
        # Alternative: 
        # For each data point i, find all j such that min_idx[j] == i.
        # Then find the min cos_d[j, i] among those.
        # This is O(N * n_test) if done naively.
        
        # Let's try a different vectorization:
        # We can compute the distance from each point to all test points? No, that's N x n_test.
        
        # Let's just use a loop over the N points. 130 iterations is fast.
        for i in range(N):
            mask = (min_idx == i)
            if np.any(mask):
                cos_vals = cos_d[mask, i]
                min_cos_i = np.min(cos_vals)
                if min_cos_i < farthest_cos[i]:
                    farthest_cos[i] = min_cos_i
                    # Get the corresponding test point
                    # We need the index of the min
                    min_idx_local = np.argmin(cos_vals)
                    global_idx = np.where(mask)[0][min_idx_local]
                    farthest_points[i] = u[global_idx]
        
        # Now update points
        # Move each point towards its farthest point
        # p_new = p + alpha * (farthest - p)
        # Then normalize
        
        # To avoid instability, we can use a smaller alpha or adaptive alpha
        # Also, if a point has no assigned test points (unlikely with 20000 test points), keep it.
        
        new_points = points.copy()
        for i in range(N):
            if farthest_cos[i] < 1.0: # If it had assigned points
                # Move towards the farthest point
                # This reduces the max distance for this cell
                new_points[i] = points[i] + alpha * (farthest_points[i] - points[i])
        
        # Normalize
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check for convergence or just update
        points = new_points
        
    return points
# EVOLVE-BLOCK-END
