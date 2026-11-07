"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses a k-Nearest Neighbors graph pruning strategy with inverse-square distance weighting.
    """
    N = distance_matrix.shape[0]
    
    # Work with a copy to avoid modifying input
    dist = distance_matrix.astype(np.float64).copy()
    
    # Ensure diagonal is infinite so nodes don't connect to themselves in k-NN selection
    np.fill_diagonal(dist, np.inf)
    
    # Define k for nearest neighbors
    k = 6
    
    # Find k nearest neighbors for each node
    # argpartition is O(N) on average, much faster than full sort
    # We want the indices of the k smallest values in each row
    k_nn_indices = np.argpartition(dist, k, axis=1)[:, :k]
    
    # Create a boolean mask for the k-NN graph (undirected)
    # An edge (i, j) is in the graph if j is in kNN(i) OR i is in kNN(j)
    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        neighbors = k_nn_indices[i]
        mask[i, neighbors] = True
        mask[neighbors, i] = True
        
    # Calculate heuristic values
    # Use inverse square distance to strongly favor short edges
    # Replace 0 with 1 to avoid division by zero (though diagonal is inf, off-diagonals are > 0)
    safe_dist = np.where(dist > 0, dist, 1.0)
    heuristic = 1.0 / (safe_dist ** 2)
    
    # Apply the mask: keep high values for k-NN edges, set others to 1e-9
    heuristic = np.where(mask, heuristic, 1e-9)
    
    # Ensure diagonal is 1e-9 (or 0, but 1e-9 is safe for probability calc)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Final cleanup for any non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
