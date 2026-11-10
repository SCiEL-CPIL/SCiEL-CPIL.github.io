"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Clarke-Wright savings heuristic for ACO CVRP.
    
    Design:
    - Base: inverse distance raised to 1.5 (stronger local preference)
    - Clarke-Wright savings: for each pair of customers i, j, compute the
      savings of merging two routes (depot->i) and (depot->j) into (depot->i->j).
      Edges with high savings get boosted, guiding ants toward distance-saving connections.
    - This targets the most beneficial merges, which is a well-known effective
      heuristic for CVRP.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base: inverse distance to the 1.5 power (stronger local preference)
    base = 1.0 / (distance_matrix + 1e-9) ** 1.5
    
    # Compute Clarke-Wright savings for all pairs of customers
    # Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    # Only valid if i != j and both are customers
    savings = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
                savings[i, j] = s
    
    # Normalize savings: clip negative savings to 0, then scale
    savings = np.maximum(savings, 0.0)
    if savings.max() > 0:
        savings = savings / savings.max()
    
    # Combine: base + savings boost
    # The savings term encourages ants to take edges that would save distance
    # in a Clarke-Wright sense
    heuristic = base + 0.8 * savings * base
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
