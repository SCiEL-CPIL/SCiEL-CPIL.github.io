"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO using Delaunay triangulation.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n, n)
    coordinates : np.ndarray
        Coordinates of shape (n, 2)
    demands : np.ndarray
        Demands of shape (n,)
    capacity : float
        Vehicle capacity
    
    Returns:
    --------
    np.ndarray
        Heuristic matrix of same shape as distance_matrix
    """
    n = len(demands)
    
    # Get customer indices (excluding depot at index 0)
    customer_indices = np.arange(1, n)
    customer_coords = coordinates[customer_indices]
    
    # Perform Delaunay triangulation on customer coordinates
    try:
        tri = Delaunay(customer_coords)
        
        # Build adjacency from triangulation
        adjacency = np.zeros((n, n), dtype=bool)
        
        # For each triangle, add edges between connected customers
        for simplex in tri.simplices:
            # simplex contains indices into customer_coords (0 to n-2)
            for i in range(3):
                for j in range(i + 1, 3):
                    ci = simplex[i] + 1  # Convert to original index
                    cj = simplex[j] + 1  # Convert to original index
                    adjacency[ci, cj] = True
                    adjacency[cj, ci] = True
        
        # Also connect depot to its nearest neighbors
        # Find nearest customers to depot
        depot_distances = distance_matrix[0, 1:n]
        # Connect depot to a few nearest customers
        k_nearest = min(10, n - 1)
        nearest_indices = np.argpartition(depot_distances, k_nearest)[:k_nearest] + 1
        for idx in nearest_indices:
            adjacency[0, idx] = True
            adjacency[idx, 0] = True
        
        # Create heuristic matrix: high weight for Delaunay edges, low for others
        # Use inverse distance for Delaunay edges, very small value for others
        heuristic = np.full((n, n), 1e-9)
        
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(heuristic, 0.0)
        
        # For edges in the Delaunay graph, use inverse distance
        for i in range(n):
            for j in range(n):
                if i != j and adjacency[i, j]:
                    dist = distance_matrix[i, j]
                    if dist > 0:
                        heuristic[i, j] = 1.0 / (dist + 1e-9)
        
        # Ensure no zero or negative values (except diagonal)
        heuristic[~np.eye(n, dtype=bool)] = np.maximum(heuristic[~np.eye(n, dtype=bool)], 1e-9)
        
    except Exception:
        # Fallback to simple inverse distance heuristic
        heuristic = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        heuristic[~np.eye(n, dtype=bool)] = np.maximum(heuristic[~np.eye(n, dtype=bool)], 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
