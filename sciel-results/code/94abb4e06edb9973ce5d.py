"""Optimized CVRP heuristic combining nearest-neighbor with depot-proximity and capacity-awareness."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot.
    
    Uses a multi-criteria scoring function:
    1. Distance from current node to candidate (primary)
    2. Distance from candidate to depot (secondary, to avoid stranding far nodes)
    3. Capacity urgency (if rest_capacity is low, prefer nodes with lower demand)
    
    The score is: dist(current, j) + alpha * dist(j, depot) + beta * (demand[j] / rest_capacity) * dist(current, j)
    where alpha and beta are tuned weights.
    """
    global _start_time, _DEADLINE
    
    # Deadline guard: if we're running out of time, just return depot to be safe
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes)
    else:
        unvisited_arr = unvisited_nodes
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    # demands is likely a 1D array where demands[i] is the demand for node i
    feasible_mask = np.zeros(len(unvisited_arr), dtype=bool)
    for i, node in enumerate(unvisited_arr):
        if demands[node] <= rest_capacity:
            feasible_mask[i] = True
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Compute distances from current_node to each feasible node
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    # Compute distances from each feasible node to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Compute demand ratio for capacity urgency
    # If rest_capacity is very low, we should prefer nodes with lower demand
    # to maximize the number of nodes we can still visit
    demand_ratios = np.array([demands[node] / rest_capacity for node in feasible_nodes])
    
    # Tune weights
    alpha = 0.3  # Weight for depot proximity
    beta = 0.2   # Weight for capacity urgency
    
    # Score: weighted combination
    # We want to minimize this
    scores = dists_from_current + alpha * dists_to_depot + beta * demand_ratios * dists_from_current
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
