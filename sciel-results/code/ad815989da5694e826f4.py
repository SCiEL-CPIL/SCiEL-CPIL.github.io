import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for the ant colony optimization.
    
    Strategy:
    Uses a heuristic that balances prize value against the "cost" of visiting a node
    and returning to the start. The cost is approximated by d[i, j] + d[j, 0].
    This discourages nodes that are far from the current position OR far from the depot,
    which is crucial for satisfying the max_len constraint.
    
    Based on observed improvements, we use a high exponent on prizes to strongly
    favor high-value nodes, and a moderate exponent on the combined distance.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Extract distance from each node j to the start node 0
    # distance_matrix[j, 0] is the distance from j to 0
    d_to_start = distance_matrix[:, 0]
    
    # Combined distance: cost to go from i to j, plus cost to return from j to start
    # This gives a better estimate of the "commitment" of visiting j
    combined_dist = distance_matrix + d_to_start[None, :]
    
    # Ensure no division by zero
    combined_dist = np.maximum(combined_dist, 1e-9)
    
    # Heuristic: (Prize^alpha) / (Combined_Distance^beta)
    # alpha=3.0 strongly differentiates high prizes from low ones.
    # beta=1.0 provides a linear penalty for the round-trip cost.
    alpha = 3.0
    beta = 1.0
    
    prize_term = np.power(prizes[None, :], alpha)
    dist_term = np.power(combined_dist, beta)
    
    H = prize_term / dist_term
    
    # Ensure all entries are positive as required
    H = np.maximum(H, 1e-9)
    
    return H
