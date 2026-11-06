"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes
    unvisited_demands = demands[unvisited]
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    # 1. Immediate distance from current
    dists_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Distance to depot (return cost proxy)
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # 3. 2-step lookahead: estimate minimum additional cost after visiting each feasible node
    # For each feasible node j, look at remaining feasible nodes (excluding j) and find
    # the min distance from j to any other feasible node, then add distance from that
    # next-best node back to depot. This approximates a 2-step route extension.
    # If no other feasible node fits, the cost is just dist(j, depot).
    
    # Build sub-matrix of distances among feasible nodes
    feas_to_feas = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    # Set diagonal to a large value so we don't pick the node itself
    np.fill_diagonal(feas_to_feas, np.inf)
    
    # For each feasible node, find the closest other feasible node
    min_next_dist = np.min(feas_to_feas, axis=1)
    
    # For each feasible node, also compute the distance from the closest next node back to depot
    # We need to know which node is closest for each
    closest_next_idx = np.argmin(feas_to_feas, axis=1)
    closest_next_nodes = feasible_nodes[closest_next_idx]
    dist_closest_to_depot = distance_matrix[closest_next_nodes, depot]
    
    # 2-step cost: dist(current, j) + min(dist(j, k) + dist(k, depot)) for k != j feasible
    # But we also need to check capacity: after visiting j, can we still visit k?
    # For simplicity in the lookahead, use a soft approximation:
    # 2-step cost = dist(j, closest_k) + dist(closest_k, depot)
    # Weight this as a lookahead term
    
    two_step_cost = min_next_dist + dist_closest_to_depot
    
    # Also consider the option of just going back to depot after j
    return_cost = dists_depot
    
    # The actual 2-step lookahead cost is the minimum of:
    # - go back to depot: dists_depot
    # - go to next feasible node: two_step_cost
    lookahead = np.minimum(return_cost, two_step_cost)
    
    # 4. Savings-based score: maximize savings = dist(current, depot) + dist(depot, j) - dist(current, j)
    # Equivalent to minimizing: dist(current, j) - dist(depot, j) - dist(current, depot)
    # But since dist(current, depot) is constant for all j, minimizing dist(current, j) - dist(depot, j)
    # is equivalent. However, this alone can be misleading.
    # Better: use the full savings concept in the score.
    
    # Combine: primary is immediate + lookahead, with savings as a tiebreaker/regularizer
    # Score = dists_current + lookahead + small_penalty * (dists_depot - dists_current)
    # The last term encourages nodes where going via current is better than going via depot
    
    scores = dists_current + 1.0 * lookahead + 0.3 * np.maximum(0, dists_depot - dists_current)
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
