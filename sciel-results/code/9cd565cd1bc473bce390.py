"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    eps = 1e-9
    
    # Structural Change:
    # Instead of subtracting a return penalty from the prize (which can zero out values
    # and lose gradient), we use the total path length increment (d[i,j] + d[j,0])
    # in the denominator. This directly models the "cost" of the move in terms of
    # the budget constraint: L + d[i,j] + d[j,0] <= max_len.
    # Nodes that are far from the depot (large d[j,0]) or far from the current node
    # (large d[i,j]) are penalized multiplicatively.
    
    # Vectorized computation of total cost for each pair (i, j)
    # d_j0 is the cost to return from j to 0. Shape (n,)
    d_j0 = distance_matrix[:, 0]
    
    # Total cost matrix: cost to go i->j and then j->0. Shape (n, n)
    # This represents the minimum additional length required if we visit j from i 
    # and then must return to 0.
    total_cost = distance_matrix + d_j0[None, :]
    
    # Ensure no division by zero
    total_cost = np.maximum(total_cost, eps)
    
    # Prize term: Normalize prizes to have mean 1 to make heuristic scale-invariant
    # This prevents the heuristic from being dominated by the absolute scale of prizes.
    mean_prize = np.mean(prizes)
    if mean_prize > 0:
        prize_norm = prizes / mean_prize
    else:
        prize_norm = prizes
        
    # Heuristic: prize_norm[j] / (total_cost[i,j])^beta
    # beta > 1 to strongly prefer low-cost moves.
    # A higher beta makes the ants more "greedy" about saving distance.
    beta = 1.5
    
    prize_term = prize_norm[None, :]  # (1, n)
    cost_term = np.power(total_cost, beta)  # (n, n)
    
    heuristic = prize_term / cost_term
    
    # Handle node 0 (start). Prize is 0, so heuristic will be 0.
    # This is correct as we shouldn't move back to start during the tour building phase 
    # (ants stop when no unvisited node qualifies, or implicitly return).
    
    # Row-normalize to have mean 1 per row.
    # This ensures that the pheromone and heuristic have a consistent scale,
    # preventing one row from having much higher probabilities than another
    # due to local geometry differences.
    row_means = np.mean(heuristic, axis=1, keepdims=True)
    row_means = np.maximum(row_means, eps)
    heuristic = heuristic / row_means
    
    # Set diagonal to very small (no self-moves)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
