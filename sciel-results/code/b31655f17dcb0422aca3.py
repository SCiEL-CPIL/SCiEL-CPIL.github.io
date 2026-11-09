import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    Structural Change:
    1. Static Feasibility Mask: Edges where d[i,j] + d[j,0] > max_len are marked as 
       permanently infeasible (heuristic = 1e-9). This is valid because path length 
       L_i >= 0.
    2. Decomposed Heuristic: Separates the 'global' value of a node (prize / dist_to_depot)
       from the 'local' transition cost (1 / dist_from_current).
       eta[i,j] = (prize[j] / d[j,0]) * (1 / d[i,j])
       This encourages visiting nodes that are both valuable relative to their distance 
       from the depot and close to the current position.
    3. Normalization: Scales the heuristic matrix so that the maximum value is 1.0.
       This stabilizes the ACO probability calculations relative to the pheromone (which 
       starts at 1.0).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # 1. Static Feasibility Mask
    # If d[i,j] + d[j,0] > max_len, then for any L_i >= 0, 
    # L_i + d[i,j] + d[j,0] > max_len. So this edge is never usable.
    effective_cost = distance_matrix + distance_matrix[:, 0:1] # d[i,j] + d[j,0]
    feasible_mask = effective_cost <= max_len
    
    # 2. Decomposed Heuristic
    # Global value: prize at j divided by distance from depot j.
    # This penalizes nodes that are far from the depot, as they consume more 
    # of the total budget just to be reached and left.
    global_value = prizes / (distance_matrix[:, 0:1] + eps) # Shape (n, 1)
    
    # Local penalty: inverse distance from current node i to j.
    # This penalizes long jumps.
    local_penalty = 1.0 / (distance_matrix + eps) # Shape (n, n)
    
    # Combine: eta[i,j] = global_value[j] * local_penalty[i,j]
    # Note: global_value is broadcast along rows.
    eta = global_value * local_penalty
    
    # Apply static feasibility mask
    # Infeasible edges get a very small value.
    eta = np.where(feasible_mask, eta, 1e-9)
    
    # Ensure node 0 is not a destination (prize is 0, so it should be low, 
    # but let's force it to be minimal to avoid any accidental attraction)
    eta[:, 0] = 1e-9
    
    # 3. Normalization
    # Find the max value to normalize.
    # If all are 1e-9, max is 1e-9.
    max_eta = np.max(eta)
    if max_eta > eps:
        eta = eta / max_eta
    else:
        # Fallback if no feasible edges? Should not happen with valid input.
        eta = np.full_like(eta, 1e-9)
        
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
