import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Fallback heuristic: standard 1/distance
    # Use alpha=2 to emphasize short edges more
    fallback = 1.0 / (distance_matrix + 1e-9) ** 2
    
    try:
        # 1. Reconstruct 2D points from distance matrix using Classical MDS
        D2 = distance_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * (J @ D2 @ J)
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # We want the top 2 eigenvalues (largest)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        top_eigvals = eigvals[:2]
        top_eigvecs = eigvecs[:, :2]
        
        # Ensure eigenvalues are non-negative
        top_eigvals = np.maximum(top_eigvals, 0)
        
        # Reconstruct points
        points = top_eigvecs * np.sqrt(top_eigvals)
        
        # 2. Compute Delaunay Triangulation
        tri = Delaunay(points)
        
        # 3. Build the heuristic matrix
        # Base heuristic: 1/d^2
        H = 1.0 / (distance_matrix + 1e-9) ** 2
        
        # Create a mask for Delaunay edges
        delaunay_mask = np.zeros((n, n), dtype=bool)
        
        # Extract unique edges from triangulation
        for simplex in tri.simplices:
            i, j, k = simplex
            delaunay_mask[i, j] = True
            delaunay_mask[j, i] = True
            delaunay_mask[j, k] = True
            delaunay_mask[k, j] = True
            delaunay_mask[i, k] = True
            delaunay_mask[k, i] = True
        
        # Apply soft constraint:
        # Delaunay edges keep full value
        # Non-Delaunay edges are penalized by a factor (e.g., 0.01)
        # This allows the ACO to use non-Delaunay edges if necessary,
        # but strongly prefers the planar subgraph.
        penalty_factor = 0.01
        H = np.where(delaunay_mask, H, H * penalty_factor)
        
        # Ensure diagonal is 0
        np.fill_diagonal(H, 0)
        
        return H
        
    except Exception:
        return fallback
