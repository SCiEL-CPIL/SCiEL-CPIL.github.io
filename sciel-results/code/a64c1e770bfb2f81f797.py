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

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Adaptive Hybrid Heuristic.
    The strategy shifts from "Nearest Neighbor" to "Closest to Depot" 
    as the remaining capacity decreases.
    
    Rationale:
    - Early in the route (high capacity), we want to pack in nearby nodes to 
      minimize travel between them. Nearest Neighbor is good here.
    - Late in the route (low capacity), we are close to closing the route. 
      We should prioritize nodes that are close to the depot to minimize 
      the final leg back to the depot, or nodes that are "on the way" 
      back to the depot.
    
    Score = (1 - alpha) * d(current, i) + alpha * d(i, depot)
    Where alpha increases as rest_capacity decreases relative to the 
    estimated total demand or max demand.
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
    
    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Estimate "progress" or "urgency".
    # We don't know the initial capacity, but we can estimate it from the max demand
    # or just use the number of remaining nodes vs total nodes?
    # A robust proxy for "how full the truck is" is not directly available.
    # However, we can use the ratio of rest_capacity to the max demand in the set?
    # Or simply: if rest_capacity is small compared to the demands of the nodes, we are "tight".
    
    # Let's use a simple heuristic for alpha:
    # If rest_capacity is very small (e.g., < 10), we are likely near the end.
    # If rest_capacity is large (e.g., > 40), we are near the start.
    # Assuming capacity is around 50 as per task description.
    
    # Normalize capacity to [0, 1] assuming max capacity ~ 50.
    # This is a guess, but robust enough for the heuristic blend.
    max_cap_estimate = 50.0
    cap_ratio = rest_capacity / max_cap_estimate
    
    # If cap_ratio is high (0.8-1.0), alpha should be low (0.0 - 0.2) -> favor NN
    # If cap_ratio is low (0.0-0.2), alpha should be high (0.6 - 1.0) -> favor Depot
    # Linear interpolation:
    # alpha = 1.0 - cap_ratio? 
    # If cap_ratio = 1.0, alpha = 0.0 (Pure NN)
    # If cap_ratio = 0.0, alpha = 1.0 (Pure Depot)
    # Let's smooth it:
    alpha = 1.0 - cap_ratio
    
    # To avoid being too aggressive towards depot early, let's clamp alpha
    # and add a small baseline depot consideration always.
    # alpha_eff = 0.2 + 0.8 * (1.0 - cap_ratio)
    alpha_eff = 0.2 + 0.8 * (1.0 - cap_ratio)
    
    # Also, if there are very few nodes left, we should definitely go to the one
    # that minimizes the total route extension.
    if n <= 3:
        # For small sets, we can afford to look ahead 1 step or just use a stronger depot bias
        # Actually, for n<=3, the difference between NN and Depot is less critical,
        # but let's stick to the blended score.
        pass
    
    # Normalize distances to make them comparable?
    # The scales of d(current, i) and d(i, depot) should be similar, so raw sum is okay.
    # But to be safe, we can normalize by the max distance in the matrix.
    max_dist = np.max(dm)
    if max_dist == 0:
        max_dist = 1.0
        
    norm_dists_current = dists_to_current / max_dist
    norm_dists_depot = dists_to_depot / max_dist
    
    scores = (1.0 - alpha_eff) * norm_dists_current + alpha_eff * norm_dists_depot
    
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
