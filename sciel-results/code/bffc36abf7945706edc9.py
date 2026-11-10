"""Improved heuristic for capacitated VRP: savings-based with capacity-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return depot (0) to close a route.
    
    Uses a savings-based heuristic with capacity-awareness:
    - Compute savings for each candidate relative to nearest neighbor
    - Penalize candidates that would make capacity tight
    - Consider the cost of returning to depot if no good candidate exists
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    n = len(unvisited_nodes)
    unvisited = np.asarray(unvisited_nodes)
    
    # Distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Distances from each unvisited to depot (for potential closure cost)
    dist_to_depot = distance_matrix[unvisited, depot]
    
    # Demands of unvisited nodes
    unvisited_demands = demands[unvisited]
    
    # Feasibility: can we actually visit this node?
    feasible_mask = unvisited_demands <= rest_capacity
    
    # If no feasible nodes, must return to depot
    if not np.any(feasible_mask):
        return depot
    
    # For feasible nodes, compute a composite score
    # Savings heuristic: cost of serving node i alone (2*dist(depot,i)) 
    # vs cost of inserting it between current and next
    # Simplified: prefer nodes that are close to current AND have good "savings"
    
    # Base score: distance from current (lower is better)
    base_score = dist_from_current.copy()
    
    # Savings component: nodes far from depot have more "savings" potential
    # when visited in sequence (reduces total depot trips)
    # savings_i = 2*dist(depot, i) - dist(current, i) - dist(i, next_best)
    # Approximate: use dist(depot, i) as proxy for savings potential
    # Higher dist_to_depot means more savings if we can chain it
    
    # Capacity tightness penalty: if visiting this node leaves very little capacity,
    # we might be forced to go back to depot soon, which is costly
    remaining_after = rest_capacity - unvisited_demands
    # Penalize if remaining capacity is small relative to average demand
    avg_demand = np.mean(demands[1:])  # exclude depot
    capacity_tightness = np.where(
        remaining_after < avg_demand,
        (avg_demand - remaining_after) / (avg_demand + 1e-9),
        0.0
    )
    
    # Composite score: minimize this
    # Weight: primarily distance, with savings bonus and capacity penalty
    # We want to minimize: dist_from_current - alpha * dist_to_depot + beta * capacity_tightness
    # But dist_to_depot being high is GOOD (more savings), so we subtract it
    
    # Actually, let me think about this differently:
    # The total route length contribution of visiting node i now is:
    # dist(current, i) + dist(i, next) + ... + dist(last, depot)
    # vs. not visiting i and going directly to depot: dist(current, depot)
    # 
    # A good heuristic: prefer nodes where dist(current, i) is small
    # and where i is far from depot (high savings potential)
    # and where capacity remains reasonable
    
    # Normalize components
    max_dist = np.max(dist_from_current) + 1e-9
    max_depot_dist = np.max(dist_to_depot) + 1e-9
    
    # Score: lower is better
    # Primary: distance from current
    # Secondary: negative savings (i.e., prefer high dist_to_depot)
    # Tertiary: capacity tightness penalty
    
    alpha = 0.3  # weight for savings
    beta = 0.5   # weight for capacity tightness
    
    score = (dist_from_current / max_dist 
             - alpha * (dist_to_depot / max_depot_dist) 
             + beta * capacity_tightness)
    
    # Mask infeasible nodes with infinity
    score = np.where(feasible_mask, score, np.inf)
    
    # Also consider: if we have few nodes left and capacity is tight,
    # it might be better to close the route now
    # Estimate cost of continuing vs closing
    cost_to_close = distance_matrix[current_node, depot]
    
    # If the best candidate's additional cost exceeds closing cost significantly,
    # and we have low remaining capacity, close the route
    best_idx = np.argmin(score)
    if np.isinf(score[best_idx]):
        return depot
    
    # Check if closing is better
    # If remaining capacity is very low (< 20% of total capacity),
    # and the best candidate is far, consider closing
    total_capacity = 50  # given in problem statement
    if rest_capacity < 0.2 * total_capacity:
        # If the best candidate would leave us with very little, and it's far
        if unvisited_demands[best_idx] > rest_capacity * 0.5:
            # Compare: cost to go to best candidate then depot vs direct depot
            cost_via_candidate = dist_from_current[best_idx] + dist_to_depot[best_idx]
            if cost_via_candidate > 1.5 * cost_to_close:
                return depot
    
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
