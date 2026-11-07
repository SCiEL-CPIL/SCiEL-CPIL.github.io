"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    # Strategy:
    # 1. Reachability cost: The cost of visiting node j from node i is d[i,j] + d[j,0].
    #    This is the exact constraint that must be satisfied for the move to be valid (given current length 0 at start, 
    #    though ants move dynamically, this static heuristic captures the "round trip" cost from start via i).
    #    Actually, the constraint is length_so_far + d[i,j] + d[j,0] <= max_len.
    #    Since length_so_far varies, a static heuristic can't perfectly capture feasibility for all states.
    #    However, nodes with high d[j,0] are generally harder to include in any route.
    #    So we penalize d[j,0] strongly.
    
    # 2. Prize amplification: Use a high exponent on prize to make high-value nodes significantly more attractive.
    #    This helps break symmetry and guides ants to "must-visit" high prize nodes early.
    
    # 3. Distance penalty: Penalize the total "commitment" distance d[i,j] + d[j,0].
    
    # Let's define the heuristic h[i,j] = prize[j]^alpha / (d[i,j] + d[j,0] + eps)^beta
    
    # Based on the hint: "Steeper prize exponent (alpha=3) and distance penalty (beta=1.5) ... improving early ant decisions"
    # Let's try alpha=3, beta=1.5.
    
    alpha = 3.0
    beta = 1.5
    eps = 1e-9
    
    # Vectorized computation
    # prizes: (N,) -> (1, N) for broadcasting against (N, N)
    p = prizes[None, :] ** alpha
    
    # cost[i, j] = distance_matrix[i, j] + distance_matrix[j, 0]
    # distance_matrix[j, 0] is the distance from j back to 0.
    return_dist = distance_matrix[:, 0] # shape (N,) -> (N, 1) for broadcasting? No, we need (1, N) or (N, N).
    # We want cost[i, j] = d[i, j] + d[j, 0]
    # d[i, j] is (N, N)
    # d[j, 0] is the 0-th column of distance_matrix, shape (N,).
    # To add to each column j of distance_matrix, we need shape (1, N).
    cost = distance_matrix + return_dist[None, :]
    
    h = p / (cost + eps) ** beta
    
    # Ensure positive
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
