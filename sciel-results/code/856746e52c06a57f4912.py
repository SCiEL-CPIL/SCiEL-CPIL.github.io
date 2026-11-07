"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of
    capacitated vehicle routing.
    
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    n = distance_matrix.shape[0]
    # Ensure we work with a proper distance matrix
    # The heuristic should encourage shorter edges
    # We use a combination of:
    # 1. Inverse distance (primary signal)
    # 2. A small penalty for edges that connect nodes with high combined demand
    #    relative to capacity (encourages grouping compatible customers)
    
    # Base heuristic: inverse distance
    # Add a tiny epsilon to avoid division by zero
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Demand-based adjustment:
    # For each edge (i, j), if demands[i] + demands[j] <= capacity, it's feasible
    # We can boost edges where the two nodes are "compatible" in a routing sense
    # but since capacity is 50 and max demand is 9, almost all pairs are compatible
    # A more subtle approach: prefer edges that don't waste capacity
    # If demands[i] + demands[j] is close to capacity, the edge is more "efficient"
    # but this is tricky because we don't know the route context
    
    # Simpler: just use inverse distance with a slight normalization
    # Scale so that the maximum heuristic value is around 1.0 for numerical stability
    max_inv = np.max(inv_dist)
    if max_inv > 0:
        heuristic = inv_dist / max_inv
    else:
        heuristic = inv_dist
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
