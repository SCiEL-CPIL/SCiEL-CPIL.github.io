"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO on TSP.
    
    Strategy: Use a combination of:
    1. Inverse distance (standard heuristic)
    2. Neighborhood structure via k-nearest neighbors to boost local edges
    3. Penalty for long-range edges that are likely to create crossings
    
    The heuristic is computed once per instance.
    """
    n = distance_matrix.shape[0]
    
    # Start with inverse distance
    # Add small epsilon to avoid division by zero
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # For the diagonal (self-loops), set to 0 since we don't visit same node
    np.fill_diagonal(inv_dist, 0.0)
    
    # Enhance with neighborhood structure:
    # For each node, identify its k nearest neighbors and boost those edges
    k = max(3, int(n * 0.1))  # 10% of nodes, at least 3
    
    # Compute row-wise sorted distances to find k-nearest neighbors
    # For each row, find indices of k smallest distances (excluding self)
    neighbor_boost = np.zeros_like(distance_matrix)
    
    for i in range(n):
        # Get distances from i to all others, excluding self
        row = distance_matrix[i].copy()
        row[i] = np.inf  # exclude self
        # Find k nearest neighbors
        k_indices = np.argsort(row)[:k]
        # Boost these edges
        for j in k_indices:
            neighbor_boost[i, j] = 1.0
            neighbor_boost[j, i] = 1.0  # symmetric boost
    
    # Combine: weight the inverse distance with a neighborhood bonus
    # Edges that are both short AND in the k-NN set get extra boost
    combined = inv_dist * (1.0 + 2.0 * neighbor_boost)
    
    # Additional refinement: penalize edges that span large distances
    # relative to the typical nearest-neighbor distance
    # Compute median nearest-neighbor distance
    nn_dists = []
    for i in range(n):
        row = distance_matrix[i].copy()
        row[i] = np.inf
        nn_dists.append(np.min(row))
    nn_dists = np.array(nn_dists)
    median_nn = np.median(nn_dists)
    
    # Scale: edges longer than ~3x median NN distance get penalized
    # Use a smooth penalty
    scale = median_nn * 3.0
    penalty = np.exp(-distance_matrix / (scale + 1e-9))
    
    # Final heuristic: combine all components
    # Base inverse distance, boosted by neighborhood, modulated by distance penalty
    heuristic = combined * (0.5 + 0.5 * penalty)
    
    # Ensure positive values; replace non-positive with 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Normalize so that the max is 1.0 (optional, helps with numerical stability)
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    return heuristic
# EVOLVE-BLOCK-END
