"""
Strategy: Dynamic Regret Minimization.
Instead of a static global tour or pure nearest-neighbor, we dynamically score candidates
based on how much their distance deviates from the average distance to all other feasible nodes.
This penalizes 'outlier' long edges, encouraging uniform segment lengths and preventing
premature capacity exhaustion on distant nodes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    if len(unvisited_nodes) == 0:
        return depot

    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Convert to numpy array for vectorized operations
    feasible_arr = np.array(feasible)
    
    # Get distances from current node to all feasible nodes
    dists = distance_matrix[current_node, feasible_arr]
    
    # Calculate the average distance to the feasible set
    # This represents the "expected" or "typical" cost of the next edge
    mean_dist = np.mean(dists)
    
    # Regret score: 
    # We want to minimize the distance, but also avoid outlier long edges.
    # The score is defined as: d(current, j) + lambda * max(0, d(current, j) - mean_dist)
    # However, the prompt suggests: "score for each candidate is the negative of the difference 
    # between the current edge cost and the average edge cost... effectively steering away from outlier-long edges"
    # Let's interpret this as minimizing a cost that penalizes deviation above the mean.
    # A simpler and robust interpretation for "steering away from outlier-long edges" is to 
    # minimize the distance, but with a penalty for being significantly larger than the mean.
    # Or, as the knowledge cell suggests: "penalizing distances above the mean".
    
    # Let's use a penalty factor. If a distance is above the mean, it adds extra cost.
    # Cost = dist + penalty * (dist - mean_dist) for dist > mean_dist, else dist.
    # This is equivalent to: dist + penalty * max(0, dist - mean_dist)
    
    # Let's try a soft penalty to make it smoother.
    # Score = dist + alpha * (dist - mean_dist)
    # This linearly penalizes any distance above the mean.
    # If alpha is large, it strongly prefers distances close to or below the mean.
    
    alpha = 1.0  # Tuning parameter for regret penalty
    
    scores = dists + alpha * (dists - mean_dist)
    
    # The candidate with the lowest score is selected
    min_idx = np.argmin(scores)
    
    return int(feasible_arr[min_idx])

# EVOLVE-BLOCK-END
