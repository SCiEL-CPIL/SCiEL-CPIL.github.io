"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for the ant colony.
    
    Structural Change:
    Instead of a quadratic penalty on distance (d_ij * (d_ij + d_j0)), 
    we use a linear incremental cost model:
    heuristic[i, j] = prize[j] / (d[i, j] + d[j, 0] + epsilon)
    
    This models the orienteering problem as a sequential selection where 
    the cost of adding node j to the current path is the sum of the distance 
    to j and the distance from j back to the origin. This encourages ants 
    to select nodes that offer the highest prize density per unit of 
    additional path length committed, balancing local accessibility 
    (d_ij) with global return efficiency (d_j0).
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Ensure prizes are non-negative
    prizes = np.maximum(prizes, 0.0)
    
    # Ensure distance matrix is non-negative and finite
    distance_matrix = np.maximum(distance_matrix, 0.0)
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    
    # d_ij: distance from i to j
    d_ij = distance_matrix  # shape (n, n)
    
    # d_j0: distance from j to 0
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    
    eps = 1e-9
    
    # prize_term: prize[j] for all i
    prize_term = prizes[None, :] + eps  # shape (1, n)
    
    # Cost term: d_ij + d_j0
    # This represents the total length added to the path if we move i -> j 
    # and then assume the return leg is d_j0. 
    # Note: In reality, the return leg is from the LAST node, but d_j0 is a 
    # good proxy for the "return cost" associated with node j in a static 
    # heuristic, especially for short paths.
    cost_term = d_ij + d_j0[None, :] + eps  # shape (n, n)
    
    # Heuristic: Prize per unit incremental cost
    heuristic = prize_term / cost_term
    
    # Optional: Scale by max_len to normalize? 
    # The magnitude of pheromone * heuristic matters for probability.
    # Since d_ij and d_j0 are in [0, ~1.4] for unit square, and max_len=3,
    # the costs are roughly 0.5 to 2.5.
    # Prizes are 0 to 1.
    # So heuristic values are roughly 0 to 2.
    # This is a reasonable scale.
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
