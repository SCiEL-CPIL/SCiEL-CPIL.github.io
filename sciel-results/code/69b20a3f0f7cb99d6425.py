import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    # Distance from each node to origin (node 0)
    dist_to_origin = distance_matrix[:, 0]
    
    # Heuristic matrix
    eta = np.zeros((n, n), dtype=np.float64)
    
    # Vectorized calculation
    # Cost of moving i -> j is d[i,j] + d[j,0]
    # We want to penalize this cost.
    # eta[i,j] = prize[j] / (cost[i,j] + eps)^alpha
    
    # Let's use alpha = 1.5
    # And handle the case where cost > max_len (infeasible even with L=0)
    
    cost = distance_matrix + dist_to_origin[None, :]
    
    # Mask for potentially feasible edges
    # If cost > max_len, it's never feasible.
    # But L >= 0, so if cost > max_len, infeasible.
    feasible_mask = cost <= max_len
    
    # Calculate heuristic
    # Avoid division by zero
    eps = 1e-9
    # We want larger eta for smaller cost and larger prize
    # eta[i,j] = prize[j] / (cost[i,j] + eps)^1.5
    
    # Only calculate for feasible
    # We can use np.where
    # But we need to broadcast prizes
    prizes_row = prizes[None, :]
    
    # Calculate raw heuristic
    raw_eta = prizes_row / (cost + eps) ** 1.5
    
    # Set infeasible to 0
    raw_eta[~feasible_mask] = 0.0
    
    # Ensure diagonal is 0 (don't move to self)
    np.fill_diagonal(raw_eta, 0.0)
    
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # So we replace 0s with 1e-9?
    # "Heuristic entries that are not positive become 1e-9"
    # This implies we should return 1e-9 for 0s?
    # Or does it mean the *input* to the ant's probability calc?
    # "Heuristic entries that are not positive become 1e-9"
    # This is a rule for the heuristic matrix returned.
    # So if I return 0, it becomes 1e-9?
    # Or I should return 1e-9 myself?
    # "Heuristic entries that are not positive become 1e-9"
    # This sounds like a post-processing step in the evaluator or a requirement for the function.
    # "write heuristics... which returns an edge-attractiveness matrix... Heuristic entries that are not positive become 1e-9"
    # It's ambiguous if *I* should do it or the *evaluator* does.
    # "The heuristic is computed once per instance... An exception, non-finite values or a wrong shape make the whole result invalid."
    # Usually, "become" implies the evaluator handles it, or I should ensure they are positive.
    # To be safe, I will replace any value <= 0 with 1e-9.
    
    raw_eta[raw_eta <= 0] = 1e-9
    
    return raw_eta
