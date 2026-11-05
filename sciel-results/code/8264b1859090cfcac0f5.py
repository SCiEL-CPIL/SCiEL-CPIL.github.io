"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_EVAL_START = time.time()
DEADLINE = _EVAL_START + 140  # Safety margin before 150s kill

def _time_remaining():
    return DEADLINE - time.time()

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a normalized savings-inspired heuristic.
    The score normalizes distances by the mean distance of the feasible set,
    which helps in steering away from outlier-long edges and maintains
    uniform segment lengths. The weight on depot proximity increases as
    remaining capacity decreases (tightness).
    """
    # Deadline guard
    if _time_remaining() < 1.0:
        # Emergency fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # If no unvisited nodes, return depot
    if len(unvisited_nodes) == 0:
        return 0

    # Filter to only feasible customers (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible = unvisited_arr[feasible_mask]

    # If no feasible customer, return depot to close route
    if len(feasible) == 0:
        return 0

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    # Distances
    d_cur = distance_matrix[current_node, feasible]
    d_dep = distance_matrix[feasible, depot]
    
    # Calculate means for normalization
    mean_d_cur = np.mean(d_cur)
    mean_d_dep = np.mean(d_dep)
    
    # Avoid division by zero (should not happen with valid distance matrices, but safe)
    if mean_d_cur == 0: mean_d_cur = 1.0
    if mean_d_dep == 0: mean_d_dep = 1.0
    
    # Normalize distances
    norm_d_cur = d_cur / mean_d_cur
    norm_d_dep = d_dep / mean_d_dep
    
    # Calculate tightness
    # Capacity is stated as 50 in the task description.
    # We use 50.0 as the reference capacity for normalization.
    max_capacity = 50.0
    # Utilization: 0 when full capacity remains, 1 when no capacity remains
    utilization = 1.0 - (rest_capacity / max_capacity)
    utilization = np.clip(utilization, 0.0, 1.0)
    
    # Weight for depot proximity
    # When capacity is full (utilization=0), we are at the start of the route.
    # We should focus on local density (low weight on depot).
    # When capacity is empty (utilization=1), we are at the end.
    # We should focus on returning to depot (high weight on depot).
    # Linear interpolation from 0.5 to 1.5
    # At utilization 0 (start): w_dep = 0.5 (favor local proximity)
    # At utilization 1 (end): w_dep = 1.5 (strongly favor depot proximity)
    w_dep = 0.5 + 1.0 * utilization
    
    # Score: Minimize (Normalized Dist to Current) - w_dep * (Normalized Dist to Depot)
    # This is equivalent to maximizing w_dep * (Norm Dist to Depot) - (Norm Dist to Current)
    # Which is a normalized savings heuristic.
    scores = norm_d_cur - w_dep * norm_d_dep
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
