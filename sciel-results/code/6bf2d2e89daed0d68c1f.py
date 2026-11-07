# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    Approach: Neighborhood Potential Density
    - For each node j, estimate the "potential" prize collectable from j, 
      assuming we arrived at j from the depot (0) with minimal cost d[0,j].
    - The potential is the sum of prizes of all nodes k that can be visited 
      from j and returned to 0 within the remaining budget (max_len - d[0,j]).
    - The heuristic for edge i->j is this potential divided by the edge cost d[i,j].
    - This rewards edges that lead to nodes which are hubs for high-value, 
      feasible neighbors.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    d = distance_matrix
    
    # Precompute return costs to depot (node 0)
    return_cost = d[:, 0]
    
    # Compute Potential for each node j
    # Potential[j] = sum of prizes[k] for all k such that d[j,k] + d[k,0] <= max_len - d[0,j]
    potential = np.zeros(n, dtype=np.float64)
    
    for j in range(n):
        # Minimum cost to reach j from 0 is d[0,j]
        # If j is 0, cost is 0, but prize is 0. 
        # We only care about j != 0 for movement, but let's compute for all.
        cost_to_j = d[0, j]
        remaining_budget = max_len - cost_to_j
        
        if remaining_budget < 0:
            potential[j] = 0.0
            continue
            
        # Feasible next nodes k from j:
        # We go j -> k -> 0.
        # Cost: d[j,k] + d[k,0]
        # Constraint: d[j,k] + d[k,0] <= remaining_budget
        feasible_next = (d[j, :] + return_cost <= remaining_budget)
        
        # Sum prizes of feasible next nodes
        # Note: This includes k=j? d[j,j]=0, so 0 + d[j,0] <= rem_budget.
        # If j is feasible to return, it's in the set. But we are already at j.
        # It doesn't hurt much, but we can exclude j to be precise about "future" prize.
        # However, the prize at j is collected when we arrive at j.
        # The heuristic guides the *choice* of j.
        # So Potential(j) should represent the value of the *rest* of the tour starting after arriving at j?
        # Or the value of the tour *including* j?
        # If we include j's prize, then H[i,j] = (Prize[j] + Future) / d[i,j].
        # If we exclude, H[i,j] = Future / d[i,j].
        # Usually, we want to reward high prize at j.
        # Let's include prizes of all feasible k, including j itself if it's feasible to return?
        # Actually, if we are at j, we have already collected Prize[j].
        # The decision is "go to j". The reward is Prize[j] + Prize[future nodes].
        # So Potential(j) should be Prize[j] + Sum(Prize[k] for k in Future).
        # But Future nodes depend on the path.
        # Let's just sum prizes of all k that are feasible to visit *after* j, plus Prize[j].
        
        # Let's refine:
        # The set of nodes we can visit in the remainder of the tour (starting from j, ending at 0)
        # must satisfy length constraints.
        # A simple approximation is the sum of prizes of all nodes k that can be visited 
        # in a direct path j->k->0 within the remaining budget.
        # This is an overestimate but captures density.
        
        potential[j] = np.sum(prizes[feasible_next])
        
    # Construct Heuristic Matrix
    H = np.zeros((n, n), dtype=np.float64)
    
    # Feasibility of edge i->j:
    # The ant is at i. It moves to j.
    # The total length so far + d[i,j] + d[j,0] <= max_len.
    # Since we don't know "length so far", we use the necessary condition:
    # d[i,j] + d[j,0] <= max_len.
    # This ensures that even if "length so far" was 0, the move is feasible.
    # If "length so far" is non-zero, it's stricter.
    # Using this necessary condition is standard for static heuristics.
    
    feasible_edges = (d + return_cost[None, :] <= max_len)
    
    epsilon = 1e-9
    
    # Calculate attractiveness for feasible edges
    # H[i,j] = Potential[j] / d[i,j]
    # If d[i,j] is 0 (i==j), we should avoid division by zero.
    # Also, if Potential[j] is 0, H is 0.
    
    safe_d = np.where(d > epsilon, d, 1.0)
    H_raw = potential[None, :] / safe_d
    
    # Apply feasibility mask
    H = np.where(feasible_edges, H_raw, epsilon)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, epsilon)
    H = np.maximum(H, epsilon)
    
    # Normalize to [epsilon, 1.0] to keep pheromone updates stable
    max_H = np.max(H)
    min_H = np.min(H)
    
    if max_H > min_H:
        H = epsilon + (H - min_H) / (max_H - min_H) * (1.0 - epsilon)
    else:
        H = np.ones_like(H) * epsilon
        
    return H
# EVOLVE-BLOCK-END
