"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of TSP.
    
    Strategy: Use a combination of inverse distance and nearest-neighbor graph structure.
    1. Compute the k-nearest neighbors graph (k=4) to identify locally important edges.
    2. Use inverse distance as the base heuristic.
    3. Boost edges that are in the k-NN graph to guide ants toward locally optimal tours.
    4. Apply a power transform to make the heuristic more discriminative for short edges.
    """
    n = distance_matrix.shape[0]
    k = 4  # Number of nearest neighbors to consider
    
    # 1. Compute k-nearest neighbors for each node
    knn_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        # Get distances from node i to all other nodes
        dists = distance_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        # Get indices of k nearest neighbors
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            knn_edges[i, j] = True
    
    # 2. Base heuristic: Inverse distance with power transform
    # Use a higher power to make short edges much more attractive
    base = 1.0 / (distance_matrix + 1e-9) ** 1.5
    
    # 3. Boost k-NN edges
    # Multiply k-NN edges by a factor to strengthen their attractiveness
    heuristic = base.copy()
    heuristic[knn_edges] *= 3.0
    
    # 4. Zero out diagonal
    np.fill_diagonal(heuristic, 0.0)
    
    # 5. Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
