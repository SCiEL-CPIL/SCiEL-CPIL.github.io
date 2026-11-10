"""CVRP Heuristic: Route-level look-ahead with nearest-neighbor completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def _complete_route_nn(current_node, depot, unvisited, rest_cap, demands, dist_mat):
    """Simulate completing the current route using nearest-neighbor with capacity check.
    
    Returns the total additional distance to complete this route and return to depot.
    """
    total_dist = 0.0
    cur = current_node
    cap = rest_cap
    remaining = list(unvisited)
    
    while remaining:
        # Find nearest feasible node
        best_node = -1
        best_dist = float('inf')
        for node in remaining:
            if demands[node] <= cap:
                d = dist_mat[cur, node]
                if d < best_dist:
                    best_dist = d
                    best_node = node
        
        if best_node == -1:
            # No feasible nodes, return to depot
            total_dist += dist_mat[cur, depot]
            break
        
        total_dist += best_dist
        cur = best_node
        cap -= demands[best_node]
        remaining.remove(best_node)
    
    # Return to depot
    total_dist += dist_mat[cur, depot]
    
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Route-level look-ahead heuristic: for each feasible candidate, simulate completing
    the current route using nearest-neighbor with capacity checks, then pick the
    candidate that minimizes total route distance.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # For each feasible candidate, simulate route completion and score by total distance
    best_node = depot  # default: close route
    best_total_dist = float('inf')
    
    # Also consider closing the route immediately
    close_dist = distance_matrix[current_node, depot]
    best_total_dist = close_dist
    
    for j in feasible_nodes:
        # Simulate: go to j, then complete route with NN
        d_to_j = distance_matrix[current_node, j]
        
        # After visiting j, remaining capacity
        new_cap = rest_capacity - demands[j]
        
        # Remaining unvisited nodes (excluding j)
        remaining = [n for n in feasible_nodes if n != j]
        
        # Complete route from j with NN
        completion_dist = _complete_route_nn(j, depot, remaining, new_cap, demands, distance_matrix)
        
        total_dist = d_to_j + completion_dist
        
        if total_dist < best_total_dist:
            best_total_dist = total_dist
            best_node = j
    
    return int(best_node)
# EVOLVE-BLOCK-END
