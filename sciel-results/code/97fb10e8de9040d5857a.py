"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit in a CVRP route.
    
    Uses a multi-criteria heuristic that balances:
    1. Distance to current node (proximity)
    2. Distance from depot (to encourage returning when far)
    3. Capacity utilization (to avoid stranding capacity)
    4. Demand of the node (to pack heavy nodes early when capacity is high)
    
    Returns 0 (depot) to close the route if no feasible node improves the score.
    """
    # Deadline guard: if we're running low on time, just return nearest feasible
    # This is a safety net; the main logic should be fast enough.
    start_time = time.time()
    
    # Convert to numpy arrays for vectorized operations
    unvisited = np.array(unvisited_nodes)
    if len(unvisited) == 0:
        return depot
    
    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get distances from depot to all unvisited
    dist_from_depot = distance_matrix[depot, unvisited]
    
    # Get demands of unvisited nodes
    node_demands = demands[unvisited]
    
    # Feasibility: demand must fit in remaining capacity
    feasible_mask = node_demands <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    # Get indices of feasible nodes
    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    feasible_dist_current = dist_from_current[feasible_indices]
    feasible_dist_depot = dist_from_depot[feasible_indices]
    feasible_demands = node_demands[feasible_indices]
    
    # If no feasible nodes, return to depot
    if len(feasible_nodes) == 0:
        return depot
    
    # Check time - if we've spent too long, just pick nearest feasible
    if time.time() - start_time > 0.01:
        return int(feasible_nodes[np.argmin(feasible_dist_current)])
    
    # Compute scores for each feasible node
    # We want to minimize total route length, so we use a weighted combination:
    # - Primary: distance from current (immediate cost)
    # - Secondary: distance from depot (future cost if we return now)
    # - Tertiary: capacity efficiency (prefer nodes that use capacity well)
    
    n_feasible = len(feasible_nodes)
    
    # Normalize distances for scaling
    max_dist_current = np.max(feasible_dist_current) if n_feasible > 0 else 1.0
    max_dist_depot = np.max(feasible_dist_depot) if n_feasible > 0 else 1.0
    
    # Weight 1: Proximity to current node (lower is better)
    # Use inverse distance to reward closer nodes
    prox_score = 1.0 / (feasible_dist_current + 1e-9)
    
    # Weight 2: Return-to-depot cost (lower is better)
    # We penalize nodes that are far from depot if we're low on capacity
    # This encourages returning to depot when we can't carry much more
    capacity_ratio = rest_capacity / 50.0  # Assuming capacity is 50
    # When capacity is low, we should prefer nodes close to depot
    # When capacity is high, we can afford to go further
    return_penalty = feasible_dist_depot * (1.0 - capacity_ratio)
    
    # Weight 3: Demand efficiency
    # Prefer nodes with higher demand when we have lots of capacity left
    # This helps pack the vehicle efficiently
    demand_score = feasible_demands * capacity_ratio
    
    # Combined score: we want to maximize this
    # - High proximity is good
    # - Low return penalty is good
    # - High demand utilization is good
    score = (2.0 * prox_score / (np.max(prox_score) + 1e-9) 
             - 1.5 * return_penalty / (np.max(return_penalty) + 1e-9)
             + 0.5 * demand_score / (np.max(demand_score) + 1e-9))
    
    # Select the node with the highest score
    best_idx = np.argmax(score)
    best_node = feasible_nodes[best_idx]
    
    # Safety check: if the best node is very far, consider returning to depot
    # This is a heuristic to avoid long detours
    if feasible_dist_current[best_idx] > 2.0 * np.median(feasible_dist_current):
        # Compare cost of going to this node vs returning to depot
        # If returning is significantly cheaper, return
        depot_dist = distance_matrix[current_node, depot]
        if depot_dist < 0.5 * feasible_dist_current[best_idx]:
            return depot
    
    return int(best_node)
# EVOLVE-BLOCK-END
