import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0
_DEADLINE = _start_time + _TIMEOUT

def _complete_route_nn(start, remaining_cap, unvisited_set, demands, dist, depot):
    # Fast NN completion
    total = 0.0
    cur = start
    cap = remaining_cap
    remaining = list(unvisited_set)
    
    while remaining:
        # Find feasible nodes
        # Optimization: Pre-compute demands array access? 
        # demands is a list/array. 
        feasible_indices = [i for i, n in enumerate(remaining) if demands[n] <= cap]
        if not feasible_indices:
            break
        
        # Get node indices
        feasible_nodes = [remaining[i] for i in feasible_indices]
        
        # Find nearest
        # dist[cur, feasible_nodes] -> array
        d = dist[cur, feasible_nodes]
        min_idx = np.argmin(d)
        best_n = feasible_nodes[min_idx]
        best_d = d[min_idx]
        
        total += best_d
        cap -= demands[best_n]
        cur = best_n
        remaining.remove(best_n)
        if cap <= 0:
            break
            
    total += dist[cur, depot]
    return total

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    dist = distance_matrix
    unvisited = list(unvisited_nodes)
    
    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return 0

    best_c = None
    best_score = np.inf

    for c in feasible:
        if time.time() > _DEADLINE:
            break
        
        d_first = dist[current_node, c]
        new_cap = rest_capacity - demands[c]
        
        # Remaining nodes for completion: all feasible except c
        # Note: Nodes infeasible now are never feasible later (cap decreases), 
        # so we only simulate over the current feasible set minus c.
        remaining_for_completion = [n for n in feasible if n != c]
        
        extra = _complete_route_nn(c, new_cap, remaining_for_completion, demands, dist, depot)
        score = d_first + extra
        
        if score < best_score:
            best_score = score
            best_c = c

    if best_c is None:
        # Fallback if time ran out
        if feasible:
            best_c = min(feasible, key=lambda n: dist[current_node, n])
        else:
            return 0

    return int(best_c)
