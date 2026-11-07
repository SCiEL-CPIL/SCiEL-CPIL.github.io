# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    Approach: Neighborhood Density/Reachability Analysis
    - Evaluates each node j based on the "potential" of other nodes reachable from j.
    - A node j is considered attractive if it allows access to many other high-value nodes
      within the remaining budget constraints.
    - Potential[j] = prize[j] + sum(prize[k] for all k != j if d[j, k] + d[k, 0] <= max_len)
    - Heuristic H[i, j] = Potential[j] / d[i, j] if edge (i, j) is feasible, else epsilon.
    - This rewards "hub" nodes that are close to many other prizes.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    d = distance_matrix
    return_cost = d[:, 0]
    epsilon = 1e-9
    
    # 1. Calculate Potential for each node j
    # Potential[j] represents the value of visiting j, which includes its own prize
    # plus the prizes of other nodes that can be visited from j (and returned to 0).
    potentials = np.zeros(n, dtype=np.float64)
    
    # Precompute feasibility of returning from any node k to 0
    # A node k is "returnable" from j if d[j, k] + d[k, 0] <= max_len
    # Note: We don't know the cost to reach j yet, but this is a necessary condition 
    # for k to be visitable *after* j.
    
    for j in range(n):
        # Base value is the prize of j
        potentials[j] = prizes[j]
        
        # Add prizes of neighbors k that are reachable from j and returnable to 0
        # Condition: d[j, k] + d[k, 0] <= max_len
        # We can vectorize this check
        costs_from_j = d[j, :]  # d[j, k] for all k
        return_costs = return_cost  # d[k, 0] for all k
        total_costs = costs_from_j + return_costs
        
        # Mask out self (j) and infeasible nodes
        mask = (total_costs <= max_len)
        mask[j] = False # Don't count self
        
        if np.any(mask):
            potentials[j] += np.sum(prizes[mask])
            
    # 2. Construct Heuristic Matrix H
    H = np.full((n, n), epsilon, dtype=np.float64)
    
    # Feasibility mask for edge i -> j
    # An edge i -> j is potentially feasible if d[i, j] + d[j, 0] <= max_len
    # This is the minimum cost to go from i to j and return to 0.
    # If this is > max_len, the edge can never be used in a valid tour.
    edge_feasible = (d + return_cost[None, :] <= max_len)
    
    # Safe division
    safe_d = np.where(d > epsilon, d, 1.0)
    
    # Calculate raw heuristic: Potential of destination / distance
    # H[i, j] = potentials[j] / d[i, j]
    H_raw = potentials[None, :] / safe_d
    
    # Apply feasibility mask
    # If edge is not feasible, keep epsilon.
    # If feasible, use H_raw.
    H = np.where(edge_feasible, H_raw, epsilon)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, epsilon)
    H = np.maximum(H, epsilon)
    
    # Normalize to [epsilon, 1.0] to keep values in a reasonable range for ACO
    max_H = np.max(H)
    min_H = np.min(H)
    
    if max_H > min_H:
        H = epsilon + (H - min_H) / (max_H - min_H) * (1.0 - epsilon)
    else:
        H = np.ones_like(H) * epsilon
        
    return H
# EVOLVE-BLOCK-END
