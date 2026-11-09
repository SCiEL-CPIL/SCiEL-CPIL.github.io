"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    # Simple cache using id, assuming matrix doesn't change
    if not hasattr(_get_dm, 'cache'):
        _get_dm.cache = {}
    key = id(distance_matrix)
    if key not in _get_dm.cache:
        _get_dm.cache[key] = np.asarray(distance_matrix)
    return _get_dm.cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def _simulate_route_length(start_node, current_node, unvisited_set, current_cap, demands, dm, depot):
    """
    Simulate the rest of the route using an improved heuristic:
    1) Nearest neighbor to build initial route
    2) Try 2-opt and insertion improvements for short routes
    Returns the total length of the route segment starting from current_node,
    visiting all feasible nodes in unvisited_set (respecting capacity),
    and ending at depot.
    """
    sim_current = current_node
    sim_cap = current_cap
    remaining = list(unvisited_set)
    route = [sim_current]
    
    # Build initial route with nearest neighbor
    while remaining:
        # Find feasible nodes
        feasible = [n for n in remaining if demands[n] <= sim_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            # Pick nearest feasible node
            dists = [dm[sim_current, n] for n in feasible]
            min_idx = np.argmin(dists)
            next_node = feasible[min_idx]
            
        sim_current = next_node
        sim_cap -= demands[next_node]
        route.append(next_node)
        remaining.remove(next_node)
        
    # Now try to improve the route with 2-opt and insertion
    # Only do this if the route is short enough (<= 15 nodes)
    if len(route) <= 16:
        route = _improve_route(route, demands, dm, depot, current_cap)
    
    # Calculate total length
    sim_len = 0.0
    for i in range(len(route) - 1):
        sim_len += dm[route[i], route[i + 1]]
    sim_len += dm[route[-1], depot]
    
    return sim_len

def _improve_route(route, demands, dm, depot, max_cap):
    """
    Improve a route using 2-opt and 2-insertion moves.
    Respects capacity constraints.
    """
    if len(route) < 3:
        return route
    
    best_route = route[:]
    best_len = _route_length(best_route, dm, depot)
    
    # Try 2-opt moves
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                # Check capacity feasibility for the reversal
                candidate = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if _check_capacity(candidate, demands, max_cap):
                    cand_len = _route_length(candidate, dm, depot)
                    if cand_len < best_len - 1e-9:
                        best_route = candidate
                        best_len = cand_len
                        improved = True
    
    # Try 2-insertion moves (remove two consecutive nodes, reinsert elsewhere)
    if len(best_route) >= 4:
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                if j == i + 1:
                    continue
                # Remove nodes at i and j
                removed = [best_route[i], best_route[j]]
                remaining_nodes = best_route[:i] + best_route[i+1:j] + best_route[j+1:]
                
                # Try reinserting at different positions
                for pos in range(1, len(remaining_nodes) + 1):
                    candidate = remaining_nodes[:pos] + removed + remaining_nodes[pos:]
                    if _check_capacity(candidate, demands, max_cap):
                        cand_len = _route_length(candidate, dm, depot)
                        if cand_len < best_len - 1e-9:
                            best_route = candidate
                            best_len = cand_len
    
    return best_route

def _route_length(route, dm, depot):
    """Calculate total route length including return to depot."""
    length = 0.0
    for i in range(len(route) - 1):
        length += dm[route[i], route[i + 1]]
    length += dm[route[-1], depot]
    return length

def _check_capacity(route, demands, max_cap):
    """Check if cumulative demand never exceeds max_cap."""
    cap = max_cap
    for node in route[1:]:  # Skip first node (current position)
        cap -= demands[node]
        if cap < 0:
            return False
    return True

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    Uses a 1-step lookahead heuristic with Nearest Neighbor simulation for the remainder.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Lookahead: For each feasible candidate, simulate the rest of the route
    # and pick the candidate that minimizes the total route length.
    # Add a density-aware penalty to encourage spatial clustering
    
    best_score = float('inf')
    best_node = depot # Default to closing if something goes wrong
    
    # Compute centroid of all feasible nodes for density scoring
    if len(feasible) > 2:
        # Use the distance matrix to approximate positions
        # We'll use a simple heuristic: score based on how well the candidate
        # groups nearby nodes together
        centroid_penalty_weight = 0.1
    else:
        centroid_penalty_weight = 0.0
    
    for v in feasible:
        # Immediate cost
        cost_to_v = dm[current_node, v]
        
        # Remaining nodes for simulation
        other_feasible = [n for n in feasible if n != v]
        new_cap = rest_capacity - demands[v]
        
        # Simulate the rest
        sim_len = _simulate_route_length(v, v, set(other_feasible), new_cap, demands, dm, depot)
        
        total_len = cost_to_v + sim_len
        
        # Add density penalty: penalize if v is far from other feasible nodes
        if centroid_penalty_weight > 0 and len(other_feasible) > 0:
            # Average distance from v to other feasible nodes
            avg_dist_to_others = np.mean([dm[v, u] for u in other_feasible])
            # Normalize by typical distance scale
            density_penalty = centroid_penalty_weight * avg_dist_to_others
            total_len += density_penalty
        
        if total_len < best_score:
            best_score = total_len
            best_node = v
            
    return int(best_node)
# EVOLVE-BLOCK-END
