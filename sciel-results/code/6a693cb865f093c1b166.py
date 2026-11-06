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
        
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    best_score = float('inf')
    best_node = -1
    
    # Pre-compute distance from current to depot (constant for this step)
    d_current_depot = distance_matrix[current_node, depot]
    
    # Vectorized distance from current to all feasible nodes
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Pre-compute distances from depot to all feasible nodes for savings
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    
    # Pre-compute demands for feasible nodes
    d_feasible = demands[feasible_nodes]
    
    # Compute savings for all feasible nodes vectorized
    savings_arr = d_current_depot + d_depot_feasible - d_current_feasible
    
    # Compute a simple 1-step lookahead: min distance from each feasible node to any other feasible node or depot
    # Use vectorized approach for efficiency
    # For each feasible node j, compute min(d(j, k) for k in feasible_nodes, k != j) or d(j, depot)
    # This is expensive, so use a cheaper proxy: just d(j, depot) as the "return cost"
    
    # Score: primarily driven by savings (maximize), with a small penalty for immediate distance
    # Higher savings = better (we want to minimize score, so negate savings)
    # Also add a small term to prefer nodes that leave more capacity (lower demand)
    # to avoid getting stuck with only high-demand nodes left
    
    # Weight savings heavily
    savings_weight = 2.0
    dist_weight = 0.5
    
    scores = dist_weight * d_current_feasible - savings_weight * savings_arr
    
    # Add a small capacity utilization bonus: prefer nodes with moderate demand
    # that don't waste too much capacity but also don't consume all of it
    # Actually, let's keep it simple: just use savings + distance
    
    # Find the best node
    best_idx = np.argmin(scores)
    best_node = int(feasible_nodes[best_idx])
    
    # Check if we should close the route: if the best savings is negative or very small
    # and the immediate distance to depot is much shorter than to any customer
    best_savings = savings_arr[best_idx]
    best_dist = d_current_feasible[best_idx]
    dist_to_depot = distance_matrix[current_node, depot]
    
    # If going to depot is clearly better (savings is negative and depot is close)
    if best_savings < 0 and dist_to_depot < best_dist * 0.8:
        return int(depot)
    
    return int(best_node)
# EVOLVE-BLOCK-END
