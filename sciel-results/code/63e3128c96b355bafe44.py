"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Score each feasible node:
    # We want to minimize total route length.
    # Heuristic: for each candidate, compute a cost that balances:
    # 1) Distance from current to candidate
    # 2) Expected remaining cost: distance from candidate to depot (if we close)
    #    or distance to next-nearest feasible (if we continue)
    # 3) Capacity penalty: prefer nodes that leave more capacity for future? 
    #    Actually, we want to pack tight to reduce number of routes.
    #    So prefer nodes with HIGHER demand (to fill capacity) but not so high
    #    that we can't fit more.
    
    # A common good heuristic for CVRP: 
    # cost = dist(current, candidate) + alpha * dist(candidate, depot) / (rest_capacity - demand + 1)
    # The second term encourages going to depot soon if capacity is low,
    # or encourages taking larger demands to fill capacity.
    
    # Let's use a weighted score:
    # score_i = dist(current, i) + beta * dist(i, depot) * (1 / (rest_capacity - demands[i] + 1))
    # Higher rest_capacity remaining after taking i means smaller denominator -> larger penalty
    # So this encourages taking nodes that leave little capacity (i.e., high demand)
    # which helps fill the vehicle.
    
    # But we also need to consider: if after taking i, no more feasible nodes exist,
    # then we must go to depot. So the "future cost" is dist(i, depot).
    # If more nodes remain, future cost is roughly dist(i, next) + ...
    
    # Simple approach: 
    # For each candidate i:
    #   remaining_cap = rest_capacity - demands[i]
    #   If no other feasible node fits in remaining_cap, future = dist(i, depot)
    #   Else, future ≈ min over j feasible with demands[j] <= remaining_cap of dist(i, j)
    #   total_est = dist(current, i) + future
    #   Pick i with min total_est
    
    # This is O(n^2) which for n=50 is fine.
    
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    demands_feasible = demands[feasible_nodes]
    
    n = len(feasible_nodes)
    scores = np.zeros(n)
    
    for idx in range(n):
        rem_cap = rest_capacity - demands_feasible[idx]
        
        # Find other feasible nodes that fit in remaining capacity
        if rem_cap <= 0:
            future_cost = dist_to_depot[idx]
        else:
            # Other nodes that can still be visited
            other_mask = demands_feasible <= rem_cap
            other_mask[idx] = False  # exclude self
            if np.any(other_mask):
                other_nodes = feasible_nodes[other_mask]
                future_cost = np.min(distance_matrix[feasible_nodes[idx], other_nodes])
            else:
                future_cost = dist_to_depot[idx]
        
        scores[idx] = dist_from_current[idx] + future_cost
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
