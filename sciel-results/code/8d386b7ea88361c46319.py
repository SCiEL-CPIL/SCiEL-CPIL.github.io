"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for ACO-CVRP.
    
    Uses a savings-enhanced inverse distance heuristic.
    The standard heuristic is 1/distance. This version incorporates
    Clarke-Wright savings, which measure the benefit of connecting two
    nodes directly rather than returning to the depot between them.
    
    Heuristic[i, j] = (1 + Savings[i, j]) / (Distance[i, j] + eps)
    """
    eps = 1e-9
    
    # Distances from the depot (node 0) to all other nodes
    d_depot = distance_matrix[0, :]
    
    # Calculate Savings matrix: S_ij = d(0, i) + d(0, j) - d(i, j)
    # Using broadcasting for efficiency
    savings = d_depot[:, None] + d_depot[None, :] - distance_matrix
    
    # Ensure savings are non-negative (Triangle inequality should guarantee this,
    # but clamping avoids negative values due to floating point precision)
    savings = np.maximum(savings, 0.0)
    
    # Compute the heuristic matrix
    # (1 + savings) boosts edges that provide significant savings
    # Division by distance maintains the preference for shorter edges
    heuristic = (1.0 + savings) / (distance_matrix + eps)
    
    # Ensure all values are positive as required by the task
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
