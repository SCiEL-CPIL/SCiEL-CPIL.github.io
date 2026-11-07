"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    The heuristic combines:
    1. Inverse distance (primary signal)
    2. Demand-aware adjustment (slight preference for lower demand nodes)
    3. Local density / nearest-neighbor bias (encourages tight clusters)
    
    Returns a matrix of the same shape as distance_matrix.
    """
    n = distance_matrix.shape[0]
    
    # Base: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Demand-aware: slightly penalize edges to high-demand customers
    # Softened penalty: 1/(1 + 0.5*d) to reduce bias against high-demand nodes
    demand_penalty = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                demand_penalty[i, j] = 1.0 / (1.0 + 0.5 * demands[j])
    
    # Local density heuristic: for each node i, boost edges to nearby nodes
    # that are also close to i's nearest neighbors (encourages local clustering)
    # Compute k-nearest neighbors for each node
    k = 5
    density_boost = np.ones((n, n))
    for i in range(n):
        # Get distances from node i to all others
        dists = distance_matrix[i, :].copy()
        dists[i] = np.inf
        # Find k nearest neighbors
        knn = np.argsort(dists)[:k]
        # Boost edges to these k nearest neighbors
        for j in knn:
            density_boost[i, j] = 1.5
            density_boost[j, i] = 1.5
    
    # Combine: base * demand_penalty * density_boost
    heuristic = base * demand_penalty * density_boost
    
    # Ensure no zeros or negative values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
