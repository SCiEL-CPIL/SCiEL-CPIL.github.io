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
    Simulate completing the route from current_node using a fast greedy heuristic
    that incorporates savings-based scoring for better route compactness.
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
            # Use savings-based scoring: prefer nodes that are close to current
            # but also have high potential to merge with other remaining nodes
            feas_arr = np.array(feasible)
            n = len(feas_arr)
            
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            
            if n > 1:
                sub_dm = dm[np.ix_(feas_arr, feas_arr)]
                # Savings: d(depot,i) + d(depot,j) - d(i,j)
                # For each i, max savings with any j != i
                savings_matrix = dists_to_depot[:, None] + dists_to_depot[None, :] - sub_dm
                np.fill_diagonal(savings_matrix, -np.inf)
                max_savings = np.max(savings_matrix, axis=1)
                
                # Normalize savings
                max_s = np.max(np.abs(max_savings)) if np.max(np.abs(max_savings)) > 0 else 1.0
                norm_savings = max_savings / max_s
            else:
                norm_savings = np.array([0.0])
            
            # Score: minimize distance to current, but reward high savings potential
            delta = 0.4
            scores = dists_to_current - delta * norm_savings
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
    
    Strategy: 1-Step Lookahead with savings-enhanced simulation.
    For each feasible candidate, estimate the total cost of the rest of the route
    if we pick that candidate now. The simulation uses a savings-based greedy
    heuristic to produce better route estimates. Choose the candidate that
    minimizes this estimate.
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
    
    for cand in feasible:
        # Create a copy of unvisited without cand
        remaining_for_sim = [n for n in unvisited if n != cand]
        
        # Simulate route starting with cand
        # The total distance will include dist(current_node, cand) + dist(cand, next) + ... + dist(last, depot)
        sim_cost = _simulate_route(cand, depot, remaining_for_sim, rest_capacity - demands[cand], demands, dm)
        total_cost = dm[current_node, cand] + sim_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand
            
    return int(best_node)
# EVOLVE-BLOCK-END
