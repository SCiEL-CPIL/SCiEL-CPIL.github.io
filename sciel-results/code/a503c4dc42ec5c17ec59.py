"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    # The constraint for moving i -> j is L_current + d[i, j] + d[j, 0] <= max_len
    # So the "cost" of the move in terms of budget is d[i, j] + d[j, 0]
    # We want high prize and low cost.
    # d[j, 0] is the distance from j to start.
    dist_to_start = distance_matrix[:, 0]
    # effective_cost[i, j] = d[i, j] + d[j, 0]
    effective_cost = distance_matrix + dist_to_start[None, :]
    
    # Heuristic: prize / effective_cost
    # Add epsilon to avoid division by zero
    heuristic = prizes[None, :] / (effective_cost + 1e-9)
    
    # Ensure positive values as per problem statement
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
