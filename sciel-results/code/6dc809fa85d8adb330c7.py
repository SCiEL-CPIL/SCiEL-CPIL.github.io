"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = None

def _get_deadline():
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 110.0
    return _deadline

def _time_exceeded():
    return time.time() > _get_deadline()

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Structural change: Use a savings-based heuristic with capacity-aware lookahead.
    
    Key insight: In CVRP, the optimal policy depends on remaining capacity.
    When capacity is tight, we should close the route sooner (prefer nodes near depot).
    When capacity is ample, we can afford to visit more nodes (prefer nearest neighbor).
    
    The score for each candidate i is:
      dist(current, i) + alpha * dist(i, depot) + beta * min_dist(i, other_unvisited)
    
    where alpha and beta are adaptive based on remaining capacity fraction.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    if _time_exceeded():
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    unvisited_arr = np.array(unvisited_nodes)
    
    # Distances from current to each feasible candidate
    d_cur = distance_matrix[current_node, feasible_arr]
    
    # Distances from each feasible candidate to depot
    d_depot = distance_matrix[feasible_arr, depot]
    
    # 1-step lookahead: min distance from each candidate to any other unvisited node
    # Vectorized: for each candidate, compute min over unvisited excluding itself
    # Shape: (len(feasible), len(unvisited))
    d_to_unvisited = distance_matrix[feasible_arr[:, np.newaxis], unvisited_arr[np.newaxis, :]]
    
    # Mask out self (where candidate == unvisited node)
    # feasible_arr[i] == unvisited_arr[j] means same node
    same_node = (feasible_arr[:, np.newaxis] == unvisited_arr[np.newaxis, :])
    d_to_unvisited_masked = np.where(same_node, np.inf, d_to_unvisited)
    
    lookahead = np.min(d_to_unvisited_masked, axis=1)
    # If no other unvisited nodes, lookahead is 0
    lookahead = np.where(np.isinf(lookahead), 0.0, lookahead)
    
    # Capacity-aware weighting:
    # Estimate how many more nodes we can visit with remaining capacity
    # Use average demand as a proxy
    avg_demand = np.mean(demands[unvisited_arr]) if len(unvisited_arr) > 0 else 1.0
    remaining_slots = rest_capacity / max(avg_demand, 1.0)
    
    # When remaining_slots is low (< 2), we're close to closing the route
    # -> weight depot distance more heavily
    # When remaining_slots is high (> 5), we have room -> weight nearest neighbor more
    if remaining_slots < 2.0:
        alpha = 0.8   # strong preference for depot proximity
        beta = 0.2    # small lookahead
    elif remaining_slots < 4.0:
        alpha = 0.4
        beta = 0.4
    else:
        alpha = 0.1   # weak depot preference
        beta = 0.7    # strong lookahead to plan ahead
    
    # Score: lower is better
    scores = d_cur + alpha * d_depot + beta * lookahead
    
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])

# EVOLVE-BLOCK-END
