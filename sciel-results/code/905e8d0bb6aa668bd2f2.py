"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    # Ensure inputs are numpy arrays for vectorized operations
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Structural Change: 
    # The baseline used Prize / Distance(i, j).
    # This version uses Prize / (Distance(i, j) + Distance(j, 0)).
    # 
    # Rationale:
    # In the Orienteering Problem, the total path length is constrained.
    # Visiting node j from i adds d[i, j] to the path. 
    # However, to complete the tour, we must return to 0, costing d[j, 0] 
    # (assuming we go directly from j to 0, or that d[j, 0] is a lower bound 
    # on the remaining cost to return).
    # Nodes that are far from the start (high d[j, 0]) are "expensive" 
    # in terms of the length budget, even if they are close to the current 
    # position i. This heuristic penalizes such nodes more heavily, 
    # guiding ants towards nodes that are both close to the current 
    # position and close to the start, which are more likely to be 
    # feasible within the max_len constraint.
    
    # Calculate the "effective cost" of moving to j from i
    # cost_ij = d[i, j] + d[j, 0]
    # d[j, 0] is the distance from node j to the start node (node 0).
    # We can extract this as a column vector.
    dist_to_start = distance_matrix[:, 0]  # Shape (N, 1)
    
    # distance_matrix is (N, N). dist_to_start is (N, 1).
    # Broadcasting: dist_to_start adds d[j, 0] to each row i for column j?
    # No, we need d[i, j] + d[j, 0].
    # distance_matrix[i, j] is d[i, j].
    # dist_to_start[j] is d[j, 0].
    # So we need to add dist_to_start (reshaped to (1, N) for columns? No, (N,1) for rows?)
    # Let's check dimensions.
    # distance_matrix[i, j]
    # We want to add d[j, 0] to each element in column j.
    # So we need a row vector of d[j, 0] for each j.
    # dist_to_start is shape (N,). 
    # If we do distance_matrix + dist_to_start, numpy broadcasts (N, N) + (N,).
    # This adds dist_to_start[k] to element [i, k]. 
    # So element [i, j] gets distance_matrix[i, j] + dist_to_start[j].
    # This is exactly d[i, j] + d[j, 0]. Correct.
    
    effective_cost = distance_matrix + dist_to_start
    
    # Heuristic: Prize / Effective Cost
    # Prize is per node j, so we broadcast prizes (N,) to (N, N) by columns.
    # prizes[j] should be added to row i? No, prize of destination j.
    # So we need a row vector of prizes for broadcasting across rows.
    # prizes is shape (N,).
    # (N, N) / (N,) broadcasts prizes to columns.
    # So heuristics[i, j] = prizes[j] / effective_cost[i, j].
    
    heuristic = prizes / (effective_cost + eps)
    
    # Ensure all values are positive as per task description
    # "Heuristic entries that are not positive become 1e-9"
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
