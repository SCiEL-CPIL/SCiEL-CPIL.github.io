"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time


def heuristics(distance_matrix):
    """
    Compute a topological adjacency heuristic using Delaunay triangulation.
    High attractiveness for edges in the triangulation, low for others.
    """
    n = distance_matrix.shape[0]
    
    # Reconstruct 2D points from distance matrix
    # Use MDS (metric multidimensional scaling) to get approximate coordinates
    # For points in unit square, we can use classical MDS
    
    # Classical MDS
    # D is the distance matrix
    D = distance_matrix.copy()
    
    # Double centering
    # J = I - (1/n) * ones
    J = np.eye(n) - np.ones((n, n)) / n
    
    # B = -0.5 * J * D^2 * J
    D_sq = D ** 2
    B = -0.5 * J @ D_sq @ J
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    # Scale by sqrt of eigenvalues
    X = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Normalize to unit square (optional, but helps)
    # Find bounding box
    min_coords = X.min(axis=0)
    max_coords = X.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords < 1e-10] = 1.0
    X = (X - min_coords) / range_coords
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(X)
    except Exception:
        # Fallback to distance-based heuristic
        return 1.0 / (distance_matrix + 1e-9)
    
    # Initialize heuristic matrix with low value
    heuristic = np.full((n, n), 1e-9)
    
    # For each triangle, add edges with high attractiveness
    for simplex in tri.simplices:
        i, j, k = simplex
        # Add all three edges
        heuristic[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
        heuristic[j, i] = 1.0 / (distance_matrix[j, i] + 1e-9)
        heuristic[j, k] = 1.0 / (distance_matrix[j, k] + 1e-9)
        heuristic[k, j] = 1.0 / (distance_matrix[k, j] + 1e-9)
        heuristic[i, k] = 1.0 / (distance_matrix[i, k] + 1e-9)
        heuristic[k, i] = 1.0 / (distance_matrix[k, i] + 1e-9)
    
    # Also add the k-nearest neighbors to ensure connectivity
    # Each node should have at least a few options
    for i in range(n):
        # Get distances to all other nodes
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        # Get top 3 nearest neighbors
        nn_indices = np.argsort(dists)[:3]
        for j in nn_indices:
            if heuristic[i, j] < 1e-8:  # If not already in triangulation
                heuristic[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
                heuristic[j, i] = 1.0 / (distance_matrix[j, i] + 1e-9)
    
    # Ensure diagonal is 0 (not used)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
