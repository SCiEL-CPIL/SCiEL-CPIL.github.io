"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix using Constraint-Aware Value Density.
    
    The heuristic favors moves to nodes with high prize and low "cost to visit and return".
    Cost is defined as d[i, j] + d[j, 0]. Moves that exceed the max_len budget are heavily penalized.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Precompute return costs: d[j, 0] for all j
    return_costs = distance_matrix[:, 0]
    
    # Initialize heuristic matrix
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Parameters for smooth penalties
    lambda_budget = 2.0  # Strength of budget penalty
    mu_distance = 1.0    # Strength of distance decay
    epsilon = 1e-9       # Small constant to avoid division by zero
    
    # Compute edge-wise heuristic values
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Total cost of moving from i to j and returning to origin
            cost = distance_matrix[i, j] + return_costs[j]
            
            # If the move exceeds the budget, keep the minimum value
            if cost > max_len:
                continue
            
            # Base value: prize-to-cost ratio
            base_value = prizes[j] / (cost + epsilon)
            
            # Smooth budget penalty: penalize moves that use a large fraction of the budget
            budget_fraction = cost / max_len
            budget_penalty = np.exp(-lambda_budget * budget_fraction)
            
            # Distance decay: favor closer nodes
            distance_decay = np.exp(-mu_distance * distance_matrix[i, j])
            
            # Combine components
            heuristic[i, j] = base_value * budget_penalty * distance_decay
    
    # Normalize the heuristic matrix to [1e-9, 1]
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    # Ensure all entries are at least 1e-9
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
