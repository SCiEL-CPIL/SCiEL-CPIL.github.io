"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def _simulate_route(current_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Simulate completing the route from current_node using a fast greedy heuristic.
    Returns the total length of the simulated route (excluding the edge to current_node's predecessor).
    """
    total_dist = 0.0
    current = current_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        # Filter feasible
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
            
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            # Fast greedy: minimize dist(current, cand) + 0.5 * min_dist(cand, others/depot)
            # This is a simplified savings/compactness heuristic for speed
            feas_arr = np.array(feasible)
            dists = dm[current, feas_arr]
            
            # Calculate a quick "isolation" penalty
            # Approximate: distance to depot + average distance to other feasible nodes
            # To be very fast, we just use distance to depot and distance to current
            # Let's stick to a very fast nearest-neighbor with a slight depot bias
            # score = dist(current, cand) + 0.2 * dist(cand, depot)
            dists_depot = dm[feas_arr, depot]
            scores = dists + 0.2 * dists_depot
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
            
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
        
    # Return to depot
    total_dist += dm[current, depot]
    
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: 1-Step Lookahead.
    For each feasible candidate, estimate the total cost of the rest of the route
    if we pick that candidate now. Choose the candidate that minimizes this estimate.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to simple greedy
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
    
    # For each feasible candidate, simulate the rest of the route
    best_cost = float('inf')
    best_node = feasible[0]
    
    # Optimization: If the number of feasible nodes is large, limit the lookahead
    # to the top K candidates based on immediate distance to save time.
    # However, with 50 customers and capacity 50, the number of nodes per route is small.
    # Let's say max route size is ~10-15. So len(feasible) is small.
    # We can afford to simulate for all feasible nodes.
    
    for cand in feasible:
        # Simulate route starting with cand
        # The simulation function needs to handle the state after picking cand
        # We pass cand as the new current node, and remove it from unvisited
        
        # Create a copy of unvisited without cand
        # Since unvisited is a list, we can use a set or just pass a filtered list
        # To avoid list copying overhead in the loop, we can pass the full list and handle it inside?
        # No, _simulate_route modifies remaining. We need a fresh list each time or careful handling.
        # Given small N, list(unvisited) is cheap.
        
        # Actually, we can optimize _simulate_route to not modify inputs if we pass a set or tuple?
        # Let's just pass a new list for safety and simplicity, N is small.
        remaining_for_sim = [n for n in unvisited if n != cand]
        
        # Note: rest_capacity for the simulation should be reduced by demands[cand]
        # because we have already "picked" cand in this hypothetical step.
        # The total distance will include dist(current_node, cand) + dist(cand, next) + ... + dist(last, depot)
        
        # Let's calculate the cost:
        # cost = dm[current_node, cand] + _simulate_route(cand, depot, remaining_for_sim, rest_capacity - demands[cand], demands, dm)
        
        sim_cost = _simulate_route(cand, depot, remaining_for_sim, rest_capacity - demands[cand], demands, dm)
        total_cost = dm[current_node, cand] + sim_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand
            
    return int(best_node)
# EVOLVE-BLOCK-END
