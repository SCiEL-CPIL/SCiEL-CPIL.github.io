"""Optimized CVRP heuristic using savings-based approach with capacity awareness."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot.
    
    Uses a savings-based scoring function:
    1. For each candidate node j, compute the 'savings' of visiting j from current
       instead of returning to depot and then going to j separately.
    2. Savings = d(current, depot) + d(depot, j) - d(current, j)
       (higher savings means more beneficial to extend current route)
    3. Also consider capacity feasibility and distance to depot for future routing.
    
    We prefer nodes with high savings (far from depot relative to current position)
    but also ensure we don't strand nodes that are far from depot.
    """
    if not unvisited_nodes or len(unvisited_nodes) == 0:
        return depot
    
    # Check time budget
    if time.time() > _DEADLINE:
        # Fallback: nearest feasible node
        feasible = [j for j in unvisited_nodes if demands[j] <= rest_capacity]
        if not feasible:
            return depot
        dists = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        j = int(unvisited_nodes[0])
        if demands[j] <= rest_capacity:
            return j
        else:
            return depot
    
    # Filter feasible nodes by capacity
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited_nodes])
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return depot
    
    if len(feasible_indices) == 1:
        return int(unvisited_nodes[feasible_indices[0]])
    
    feasible_nodes = unvisited_nodes[feasible_indices]
    
    # Compute scores for each feasible candidate
    # Savings-based: d(current, depot) + d(depot, j) - d(current, j)
    # Higher savings = more beneficial to visit j now rather than in a separate route
    
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_candidates = distance_matrix[depot, feasible_nodes]
    d_current_candidates = distance_matrix[current_node, feasible_nodes]
    
    # Savings: how much we save by extending current route to j instead of starting new route
    savings = d_current_depot + d_depot_candidates - d_current_candidates
    
    # We want to maximize savings, but also consider:
    # 1. The demand of the node (prefer lower demand if capacity is tight)
    # 2. Distance from candidate to depot (prefer nodes closer to depot to avoid stranding)
    
    demands_candidates = np.array([demands[j] for j in feasible_nodes])
    
    # Capacity urgency: if rest_capacity is low, penalize high demand
    capacity_ratio = rest_capacity / 50.0  # Assuming total capacity is 50
    demand_penalty = demands_candidates * (1.0 - capacity_ratio) * 2.0
    
    # Depot proximity: penalize nodes that are far from depot (to avoid stranding)
    depot_distance_penalty = d_depot_candidates * 0.3
    
    # Combine: maximize savings, minimize penalties
    score = savings - demand_penalty - depot_distance_penalty
    
    # Pick the node with highest score
    best_idx = np.argmax(score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
