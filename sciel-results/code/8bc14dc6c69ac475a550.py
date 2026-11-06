# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return depot if too much time has passed
    if time.time() - _start_time > 145:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter by capacity feasibility
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited[feasible_mask]
    
    # Score each feasible node: incremental cost = dist(current, node) + dist(node, depot)
    # We want to minimize this
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    scores = dists_from_current + dists_to_depot
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
