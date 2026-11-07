import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness heuristic matrix for ant colony optimization
    of the orienteering problem.
    
    The heuristic is based on the efficiency of visiting a node j from node i,
    considering both the travel cost d[i, j] and the return cost d[j, 0].
    Efficiency = prize[j] / (d[i, j] + d[j, 0])
    
    We raise this to a power to sharpen the difference between good and bad edges.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    N = len(prizes)
    
    # Calculate the total cost of going from i to j and then returning to 0.
    # distance_matrix[i, j] is d[i, j]
    # distance_matrix[j, 0] is d[j, 0]
    # We need a matrix where element [i, j] is d[i, j] + d[j, 0]
    return_costs = distance_matrix[:, 0]  # Shape (N,)
    total_cost = distance_matrix + return_costs[None, :]  # Shape (N, N)
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    safe_total_cost = total_cost + eps
    
    # Calculate base efficiency: Prize per unit of total cost
    # prizes[None, :] has shape (1, N)
    # safe_total_cost has shape (N, N)
    base_efficiency = prizes[None, :] / safe_total_cost
    
    # Apply a power to amplify the signal. 
    # A power > 1 favors high-efficiency edges more strongly.
    # A power < 1 would flatten the distribution.
    # Let's try 1.5 or 2.0. 1.0 is the baseline ratio.
    power = 1.5
    heuristic = base_efficiency ** power
    
    # Handle any non-finite values (shouldn't happen with eps, but just in case)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Enforce positivity as per task description
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
