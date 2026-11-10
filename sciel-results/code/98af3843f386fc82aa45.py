"""Route-level look-ahead heuristic for Capacitated Vehicle Routing Problem.

Instead of scoring each candidate with a myopic combination of savings /
nearest-neighbor / demand-fit, we simulate the *completion* of the current
route under a fast nearest-neighbor-with-capacity policy for every feasible
candidate, and pick the candidate that minimizes the predicted total length
of the current route (already-traveled distance + predicted completion).
This is a different design family: a route-level surrogate objective with a
cheap constructive completion model, rather than a step-level greedy score.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def _complete_route_nn(start, remaining_cap, unvisited_set, demands, dist,
                       max_steps=30):
    """
    Simulate completion of a route starting at `start` with `remaining_cap`
    capacity, visiting nodes from `unvisited_set` using nearest-neighbor
    with capacity constraint. Returns the total additional distance.
    """
    total_dist = 0.0
    current = start
    cap = remaining_cap
    visited_local = set()
    
    for _ in range(max_steps):
        # Find feasible unvisited nodes
        feasible = []
        for node in unvisited_set:
            if node not in visited_local and demands[node] <= cap:
                feasible.append(node)
        
        if not feasible:
            break
        
        # Pick the nearest feasible node
        best_node = None
        best_dist = float('inf')
        for node in feasible:
            d = dist[current, node]
            if d < best_dist:
                best_dist = d
                best_node = node
        
        if best_node is None:
            break
        
        total_dist += best_dist
        visited_local.add(best_node)
        cap -= demands[best_node]
        current = best_node
    
    # Add distance back to depot
    total_dist += dist[current, 0]
    
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using route-level look-ahead:
    For each feasible candidate, simulate the rest of the route using
    nearest-neighbor-with-capacity, and pick the candidate that yields
    the shortest predicted total route length.
    Returns 0 (depot) if no feasible node can be added or if time limit is reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # For each feasible candidate, simulate route completion
    # and compute predicted total route length
    
    # We need to track which nodes are already visited (not in unvisited set)
    unvisited_set = set(unvisited_nodes)
    
    best_score = float('inf')
    best_node = 0  # default: go back to depot
    
    # Pre-compute: distance from current to depot (baseline if we close now)
    d_cur_dep = distance_matrix[current_node, depot]
    
    # For each feasible candidate, simulate
    for cand in feasible:
        # Check time
        if time.time() > _DEADLINE:
            break
        
        cand = int(cand)
        
        # After visiting cand, remaining capacity and unvisited set
        new_cap = rest_capacity - demands[cand]
        new_unvisited = unvisited_set - {cand}
        
        # Simulate completion from cand
        completion_dist = _complete_route_nn(
            cand, new_cap, new_unvisited, demands, distance_matrix
        )
        
        # Total predicted route length:
        # (distance already traveled is constant for all candidates, so we
        #  can ignore it) + distance from current to cand + completion
        d_to_cand = distance_matrix[current_node, cand]
        
        total_predicted = d_to_cand + completion_dist
        
        if total_predicted < best_score:
            best_score = total_predicted
            best_node = cand
    
    return int(best_node)

# EVOLVE-BLOCK-END
