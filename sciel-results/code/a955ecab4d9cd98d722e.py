import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    # prizes: shape (N,)
    # distance_matrix: shape (N, N)
    # max_len: float
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # d_j0 is the distance from node j to the depot (node 0)
    # We need to add this to d_ij for each pair (i, j)
    # distance_matrix[:, 0] has shape (N,), which is d_j0 for each j
    # We want a matrix where entry (i, j) is d_ij + d_j0
    # distance_matrix is (N, N) where row i, col j is d_ij
    # distance_matrix[:, 0][:, None] is (N, 1) where row j is d_j0
    # Broadcasting: (N, N) + (N, 1) -> (N, N) where (i, j) is d_ij + d_j0
    
    effective_cost = distance_matrix + distance_matrix[:, 0][:, None]
    
    # Avoid division by zero or issues with 0 cost (diagonal might be 0)
    # The problem states ants move to unvisited nodes. i != j.
    # But we must return a full matrix.
    # If effective_cost is 0, the heuristic should be high? 
    # But d_ii = 0, d_i0 is not 0 usually. 
    # Wait, d_ii = 0. effective_cost[i, i] = 0 + d_i0.
    # If i=0, effective_cost[0,0] = 0.
    # We should handle zeros.
    
    # Add a small epsilon to prevent division by zero
    eps = 1e-9
    safe_cost = effective_cost + eps
    
    # Heuristic formula:
    # eta_ij = (prize_j / safe_cost_ij^gamma) * exp(-lambda * safe_cost_ij / max_len)
    
    gamma = 1.5
    lambda_decay = 1.0
    
    # prize_j is broadcasted across rows
    prize_factor = prizes[None, :] / (safe_cost ** gamma)
    
    # Exponential penalty for budget consumption
    budget_penalty = np.exp(-lambda_decay * safe_cost / max_len)
    
    heuristic = prize_factor * budget_penalty
    
    # Ensure no negative or non-finite values (though formula produces positive for positive inputs)
    # The evaluator replaces non-positive with 1e-9, but we can be safe.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
