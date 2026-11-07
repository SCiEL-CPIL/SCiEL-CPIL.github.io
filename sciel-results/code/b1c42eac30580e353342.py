"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ant colony optimization.
    
    Strategy: 
    Uses a "budget-aware" potential. The attractiveness of moving from i to j 
    is driven by the prize of j, but discounted by the total path cost 
    (dist(i,j) + dist(j,0)). This ensures ants prefer nodes that are not only 
    close and valuable, but also leave enough "budget" to return to the start, 
    which is a hard constraint in the orienteering problem.
    
    Formula: H[i, j] = (prize_j + epsilon) / (dist(i,j) + dist(j,0) + epsilon)^alpha
    We use alpha=1.5 to sharpen the preference for high-value, low-total-cost moves.
    """
    n = len(prizes)
    # Ensure inputs are numpy arrays for vectorization
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Total cost to visit j and return to 0
    # dist(j, 0) is the return leg. dist(i, j) is the travel leg.
    # Note: distance_matrix[i, j] is the cost from i to j.
    # We want H[i, j] to reflect the value of j relative to the cost of reaching it and returning.
    
    # Return cost for each node j (distance from j to 0)
    return_cost = distance_matrix[:, 0]  # Shape (n,)
    
    # Create a matrix of return costs: R[i, j] = return_cost[j]
    # We want to add this to distance_matrix[i, j]
    # distance_matrix is (n, n)
    # return_cost is (n,)
    # So R = distance_matrix + return_cost[None, :]
    total_cost_matrix = distance_matrix + return_cost[None, :]
    
    # Avoid division by zero
    epsilon = 1e-9
    
    # Prize matrix: P[i, j] = prizes[j]
    # We want to attract towards high prizes.
    prize_matrix = prizes[None, :]
    
    # Heuristic: Prize / (Total Cost)^alpha
    # Using alpha > 1 penalizes expensive moves more heavily, which is crucial 
    # for tight max_len constraints.
    alpha = 1.5
    
    # Calculate heuristic
    # If total cost is very small, heuristic is large.
    # If prize is 0, heuristic is small (but we add epsilon to prize to avoid 0s if needed, 
    # though the prompt says non-positive becomes 1e-9 later, so 0 is fine here as it will be handled).
    # However, the prompt says "Heuristic entries that are not positive become 1e-9".
    # So we can just compute the raw value.
    
    heuristic = (prize_matrix + epsilon) / (total_cost_matrix + epsilon) ** alpha
    
    # Ensure no negative or zero values cause issues later, though the evaluator handles it.
    # But we should ensure the shape is correct.
    
    return heuristic
# EVOLVE-BLOCK-END
