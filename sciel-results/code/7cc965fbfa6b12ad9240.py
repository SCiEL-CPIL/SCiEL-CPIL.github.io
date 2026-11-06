"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 150s limit, stop at 145s
    if time.time() - _start_time > 145:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    d_current_to_nodes = distance_matrix[current_node, feasible_nodes]
    d_nodes_to_depot = distance_matrix[feasible_nodes, depot]
    d_current_to_depot = distance_matrix[current_node, depot]
    
    # Clarke-Wright savings with tuned beta
    # effective_cost = d_current_node*(1+beta) - beta*(d_current_depot + d_node_depot)
    beta = 0.7
    effective_cost = d_current_to_nodes * (1.0 + beta) - beta * (d_current_to_depot + d_nodes_to_depot)
    
    # Smooth capacity-aware penalty:
    # As remaining capacity after visiting drops, add a penalty that grows
    # quadratically to discourage choices that leave too little room
    remaining_after = rest_capacity - demands[feasible_nodes]
    min_demand = float(np.min(demands))
    
    # Normalize remaining capacity: 0 means no capacity left, 1 means plenty
    # Use a smooth penalty that increases as remaining_after approaches min_demand
    # penalty = gamma * max(0, (min_demand - remaining_after + margin))^2 / scale
    # where margin provides a buffer zone
    margin = 1.0  # start penalizing when remaining < min_demand + margin
    scale = float(max(1.0, min_demand * 2.0))
    deficit = np.maximum(0.0, min_demand + margin - remaining_after)
    capacity_penalty = 0.3 * (deficit ** 2) / scale
    
    effective_cost = effective_cost + capacity_penalty
    
    # If we're the last few nodes or capacity is very tight, strongly prefer depot
    # to avoid getting stranded with infeasible remaining nodes
    if n_unvisited <= 3 or rest_capacity < 2.0 * min_demand:
        # Add a bonus to go to depot if capacity is critically low
        if np.min(remaining_after) < min_demand:
            # Can't visit any more nodes after this one, so just pick closest
            effective_cost = d_current_to_nodes.copy()
    
    best_idx = int(np.argmin(effective_cost))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
