"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Uses a combination of:
    1. Inverse distance (basic heuristic)
    2. Angle-based penalty: penalize edges that would create sharp turns
       relative to the local neighborhood structure
    3. Degree-based adjustment: slightly favor edges that connect to nodes
       with higher local connectivity in a geometric sense
    
    The key insight: in a good TSP tour, edges should not create unnecessary
    crossings or sharp angles. We can approximate this by looking at the
    local geometry around each node.
    """
    n = distance_matrix.shape[0]
    D = distance_matrix.astype(np.float64)
    
    # Basic inverse distance
    inv_dist = 1.0 / (D + 1e-9)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Compute local neighborhood structure
    # For each node, find its k-nearest neighbors
    k = min(10, n - 1)
    knn = np.argsort(D, axis=1)[:, :k]
    
    # Build a local angle penalty
    # For each pair (i, j), if j is not among i's k-nearest neighbors,
    # apply a mild penalty. This encourages ants to consider geometrically
    # plausible edges.
    # Create a matrix where entry (i,j) = 1 if j is in i's k-NN, else 0
    knn_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        knn_mask[i, knn[i]] = 1.0
    
    # Symmetrize the mask (if either i is in j's k-NN or j is in i's k-NN)
    sym_mask = np.maximum(knn_mask, knn_mask.T)
    
    # The heuristic: boost edges that are in the k-NN neighborhood
    # Use a smooth blend: base * (1 + alpha * knn_indicator)
    alpha = 0.5
    heuristic = inv_dist * (1.0 + alpha * sym_mask)
    
    # Additionally, apply a mild penalty for very long edges relative to local scale
    # Compute local distance scale for each node (mean distance to k-NN)
    local_scale = np.zeros(n)
    for i in range(n):
        local_scale[i] = np.mean(D[i, knn[i]])
    
    # For each edge (i,j), compare D[i,j] to the geometric mean of local scales
    # If D[i,j] is much larger than both local scales, penalize
    local_scale_outer = np.sqrt(np.outer(local_scale, local_scale) + 1e-9)
    ratio = D / (local_scale_outer + 1e-9)
    
    # Smooth penalty: exp(-beta * max(0, ratio - 1)^2)
    beta = 0.3
    excess = np.maximum(0.0, ratio - 1.0)
    penalty = np.exp(-beta * excess ** 2)
    
    heuristic = heuristic * penalty
    
    # Ensure diagonal is 0 (or very small, but it won't be used)
    np.fill_diagonal(heuristic, 0.0)
    
    # Clamp to positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
