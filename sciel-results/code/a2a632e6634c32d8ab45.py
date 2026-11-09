import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    Design:
    1. Static Feasibility Mask: Edges where d[i,j] + d[j,0] > max_len are marked as 
       permanently infeasible (heuristic = 1e-9).
    2. Combined Heuristic:
       - Global value: prize[j] / d[j,0] (rewards valuable nodes that are close to depot)
       - Local penalty: 1 / d[i,j] (rewards close transitions)
       - Effective cost awareness: 1 / (d[i,j] + d[j,0]) (rewards efficient round-trips)
       
       The heuristic is a product of these factors, emphasizing nodes that are 
       valuable, close to the current position, and efficient to visit and return from.
    3. Normalization: Scales the heuristic matrix so that the maximum value is 1.0.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # 1. Static Feasibility Mask
    # If d[i,j] + d[j,0] > max_len, then for any L_i >= 0, 
    # L_i + d[i,j] + d[j,0] > max_len. So this edge is never usable.
    effective_cost = distance_matrix + distance_matrix[:, 0:1]  # d[i,j] + d[j,0]
    feasible_mask = effective_cost <= max_len
    
    # 2. Combined Heuristic
    # Global value: prize at j divided by distance from depot j.
    global_value = prizes / (distance_matrix[:, 0:1] + eps)  # Shape (n, 1)
    
    # Local penalty: inverse distance from current node i to j.
    local_penalty = 1.0 / (distance_matrix + eps)  # Shape (n, n)
    
    # Effective cost awareness: inverse of total round-trip cost
    effective_cost_penalty = 1.0 / (effective_cost + eps)  # Shape (n, n)
    
    # Combine all three factors
    # eta[i,j] = global_value[j] * local_penalty[i,j] * effective_cost_penalty[i,j]
    eta = global_value * local_penalty * effective_cost_penalty
    
    # Apply static feasibility mask
    eta = np.where(feasible_mask, eta, 1e-9)
    
    # Ensure node 0 is not a destination
    eta[:, 0] = 1e-9
    
    # 3. Normalization
    max_eta = np.max(eta)
    if max_eta > eps:
        eta = eta / max_eta
    else:
        eta = np.full_like(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return eta
    
    return eta
