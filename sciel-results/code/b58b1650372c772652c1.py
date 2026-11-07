# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a demand-aware, multi-criteria edge-attractiveness matrix for ACO CVRP.
    
    Uses: eta[i,j] = d(i,j)^-2 * (1 / (1 + alpha * demand[j]))
    with alpha tuned to balance distance and demand packing.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    
    # Inverse distance squared
    eps = 1e-9
    dist = distance_matrix + eps
    inv_dist_sq = 1.0 / (dist * dist)
    
    # Demand-aware factor: stronger penalty for high-demand nodes
    # This encourages packing heavier customers early to reduce trips
    alpha = 1.0
    demand_factor = 1.0 / (1.0 + alpha * demands[np.newaxis, :])
    
    # Combine
    eta = inv_dist_sq * demand_factor
    
    # Ensure all values are positive and finite
    eta = np.maximum(eta, 1e-9)
    
    # Handle diagonal
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
