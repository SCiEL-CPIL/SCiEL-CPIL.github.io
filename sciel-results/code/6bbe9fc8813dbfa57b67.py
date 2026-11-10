"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    Uses an additive cost model based on the total length consumed by visiting
    node j from i and returning to the start. This directly reflects the
    constraint on the total path length.
    
    Formula: eta[i, j] = prize[j] / (distance[i, j] + distance[j, 0] + epsilon)
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Ensure shapes are compatible
    n = prizes.shape[0]
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-9
    
    # distance[j, 0] is the cost to return to start from j
    return_cost = distance_matrix[:, 0]  # Shape (n,)
    
    # Additive cost: sum of travel distance and return distance
    # This represents the total additional length added to the path if we
    # go from i to j and then from j back to 0 (conceptually).
    # Note: In the actual path, we don't go j->0 immediately, but the
    # constraint checks L + d[i,j] + d[j,0] <= max_len. So this cost
    # aligns with the feasibility check.
    additive_cost = distance_matrix + return_cost[None, :] + epsilon
    
    # Heuristic: Prize / Additive Cost
    heuristic = prizes[None, :] / additive_cost
    
    # Handle any non-finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
