"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO.
    
    The heuristic combines:
    1. Prize density: prize[j] / (dist[i,j] + dist[j,0])
    2. Node Centrality: Inverse of the sum of distances from node j to all other nodes.
       This biases the search towards dense clusters of nodes, which are more
       efficient to traverse within the max_len constraint.
    """
    start_time = time.time()
    n = len(prizes)
    
    # 1. Calculate return distances from each node to the depot (node 0)
    # distance_matrix[:, 0] gives dist[j, 0] for all j
    return_dist = distance_matrix[:, 0]
    
    # 2. Calculate Node Centrality
    # Centrality is defined as the inverse of the sum of distances to all other nodes.
    # Nodes in clusters have lower sum of distances -> higher centrality.
    # We add epsilon to avoid division by zero (though dist[j,j]=0, sum of others > 0 for n>1)
    eps = 1e-9
    
    # Sum of distances from each node j to all other nodes k
    # distance_matrix is n x n.
    # sum_dist[j] = sum(distance_matrix[j, :])
    sum_dist = np.sum(distance_matrix, axis=1)
    
    # Centrality score for each node
    # Higher centrality for nodes closer to others (cluster centers)
    centrality = 1.0 / (sum_dist + eps)
    
    # 3. Construct the Heuristic Matrix H
    # H[i, j] represents the attractiveness of moving from i to j.
    # Base term: prize[j] / (dist[i, j] + dist[j, 0])
    # Enhanced term: prize[j] * centrality[j] / (dist[i, j] + dist[j, 0])
    
    # Numerator: prize[j] * centrality[j]
    # Shape (n,) -> broadcast to (n, n) along axis 0
    numer = (prizes * centrality)
    
    # Denominator: dist[i, j] + dist[j, 0]
    # dist[j, 0] is return_dist[j], shape (n,)
    # We need to add return_dist[j] to each element in column j.
    # distance_matrix is (n, n).
    # return_dist[None, :] is (1, n), so it broadcasts to (n, n) adding to each column.
    denom = distance_matrix + return_dist[None, :] + eps
    
    H = numer[None, :] / denom
    
    # Ensure all values are positive and finite
    H = np.maximum(H, 1e-9)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Safety check for time (though this computation is O(N^2) and very fast for N=50)
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback to a simple heuristic if somehow slow (unlikely)
        simple_numer = prizes + 1e-9
        simple_denom = distance_matrix + return_dist[None, :] + 1e-9
        H = simple_numer[None, :] / simple_denom
        H = np.maximum(H, 1e-9)
        
    return H
# EVOLVE-BLOCK-END
