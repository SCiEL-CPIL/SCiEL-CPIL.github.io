"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for the ant colony.
    
    The heuristic encourages moves to nodes j that:
    1. Have high prize value
    2. Are close to the current node i (low d[i,j])
    3. Allow the ant to still return to node 0 (low d[j,0])
    4. Fit within the max_len constraint
    
    We use: heuristic[i,j] = prize[j] / (d[i,j] * (d[i,j] + d[j,0]) + eps)
    This rewards high prizes at nodes that are both close to i and allow return to 0.
    
    We also consider the remaining budget: since max_len is small (3), nodes with
    high d[j,0] are less attractive because they consume more of the budget for return.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Ensure prizes are non-negative
    prizes = np.maximum(prizes, 0.0)
    
    # Ensure distance matrix is non-negative and finite
    distance_matrix = np.maximum(distance_matrix, 0.0)
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e6)
    
    # Compute heuristic: for each edge (i, j), attractiveness based on:
    # - prize at node j (higher is better)
    # - distance from i to j (lower is better)
    # - distance from j back to 0 (lower is better, since it must fit in budget)
    
    # d[i,j] is distance_matrix[i,j]
    # d[j,0] is distance_matrix[j,0]
    
    d_ij = distance_matrix  # shape (n, n)
    d_j0 = distance_matrix[:, 0]  # shape (n,) - distance from each node j to node 0
    
    # For node 0, we don't want ants to "go" to node 0 mid-route (they start there)
    # But the heuristic is used for all edges; node 0's prize is 0 anyway.
    
    # Remaining budget consideration: max_len - d[i,j] - d[j,0] should be >= 0
    # We penalize edges where d[i,j] + d[j,0] is close to or exceeds max_len
    
    # Cost of using edge (i,j): d[i,j] for the move, then d[j,0] for return
    # Total "commitment" from this move: d[i,j] + d[j,0]
    # If this exceeds max_len, the move is infeasible (from start), but from mid-route
    # the remaining budget is less. A static heuristic can't capture this perfectly.
    
    # Strategy: 
    # numerator: prize[j] (broadcast across rows)
    # denominator: d[i,j] * (d[i,j] + d[j,0])
    #   - This penalizes both long moves and nodes far from origin
    #   - The product form makes it sensitive to both factors
    
    eps = 1e-9
    prize_term = prizes[None, :] + eps  # shape (1, n)
    d_ij_safe = d_ij + eps  # shape (n, n)
    d_j0_safe = d_j0[None, :] + eps  # shape (1, n)
    
    # Denominator: d[i,j] * (d[i,j] + d[j,0])
    denom = d_ij_safe * (d_ij_safe + d_j0_safe)
    
    heuristic = prize_term / denom
    
    # Additionally, we can boost nodes that are "reachable" within max_len from any node
    # i.e., d[i,j] + d[j,0] < max_len for some i. But since this is static, let's
    # add a penalty for nodes where d[j,0] is large relative to max_len.
    
    # Penalty factor: 1 / (1 + d[j,0] / max_len)
    # This reduces attractiveness of nodes far from origin
    if max_len > 0:
        penalty = 1.0 / (1.0 + d_j0_safe / max_len)  # shape (1, n)
        heuristic = heuristic * penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
