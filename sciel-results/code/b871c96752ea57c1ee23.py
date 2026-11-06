"""Rewritten program for capacitated vehicle routing heuristic selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s limit


def _elapsed_time():
    return time.time() - _DEADLINE_START


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return depot (0) to close a route.
    
    Strategy: Regret-based selection combined with nearest-neighbor fallback.
    For each feasible customer, compute the regret (difference between the
    best and second-best insertion cost). Prioritize customers with high regret
    (i.e., those that are uniquely well-positioned and would suffer large
    detours if delayed).
    
    Additionally, we use a soft scoring that balances:
    1. Immediate distance (proximity to current node)
    2. Regret (opportunity cost of not visiting this node now)
    3. Demand utilization (prefer nodes that help exhaust capacity efficiently)
    """
    if len(unvisited_nodes) == 0:
        return 0
    
    # Check if we should return to depot
    if rest_capacity < min(demands[unvisited_nodes]):
        return 0
    
    # If only one node left, go to it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Convert to arrays for vectorized operations
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Get distances from current node to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_arr]
    
    # Get demands of unvisited nodes
    demands_unvisited = demands[unvisited_arr]
    
    # Filter feasible nodes (demand <= rest_capacity)
    feasible_mask = demands_unvisited <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0
    
    if len(feasible_indices) == 1:
        return int(unvisited_arr[feasible_indices[0]])
    
    # Compute regret for each feasible node
    # Regret = distance to node - distance to second-best alternative
    # We approximate "insertion cost" as the cost of going from current to node
    # and then from node back to depot (or the best remaining connection)
    
    # For efficiency, we compute a simpler regret:
    # For each feasible node i, look at the distances from i to all other feasible nodes
    # and to the depot. The regret is how much worse it would be to delay visiting i.
    
    # Simplified approach: 
    # Score = alpha * (distance from current) + beta * (regret) + gamma * (demand penalty)
    
    # Compute regret: for each feasible node, find the minimum distance to any other
    # feasible node or depot, and compare to its own distance from current.
    
    # Get submatrix of distances among feasible nodes
    feasible_node_ids = unvisited_arr[feasible_indices]
    n_feasible = len(feasible_node_ids)
    
    if n_feasible <= 2:
        # Just pick the nearest
        return int(unvisited_arr[feasible_indices[np.argmin(dists_from_current[feasible_indices])]])
    
    # Distance matrix among feasible nodes
    dist_feasible = distance_matrix[np.ix_(feasible_node_ids, feasible_node_ids)]
    
    # Distance from feasible nodes to depot
    dist_to_depot = distance_matrix[feasible_node_ids, depot]
    
    # For each feasible node i, compute:
    # - d_i = distance from current to i
    # - For regret: consider that if we don't visit i now, we might have to go 
    #   from some other node j to i later. The "regret" is approximately
    #   how isolated i is compared to others.
    
    # A simpler and effective heuristic: 
    # Score each node by: dist_from_current[i] - lambda * (min distance from i to any other feasible node or depot)
    # Nodes that are close to current but far from others (isolated) get higher priority.
    
    # Actually, let's use a different approach: 
    # For each feasible node i, compute the cost of inserting i into the route.
    # Since we're doing nearest-neighbor style, the insertion cost is just dist_from_current[i].
    # But we want to prioritize nodes that would be expensive to visit later.
    
    # Regret-based: For node i, regret = min_j(dist(i,j)) where j is in feasible set or depot,
    # but we want nodes that have HIGH minimum distance to alternatives (they're isolated),
    # combined with LOW distance from current.
    
    # Let's compute for each feasible node:
    # 1. dist_from_current[i]
    # 2. min_dist_to_others[i] = min over j != i of dist(i,j), and dist(i, depot)
    
    # For efficiency with n_feasible up to 50, this is fine
    min_dist_to_alternatives = np.empty(n_feasible)
    for i in range(n_feasible):
        # Distance to all other feasible nodes
        if n_feasible > 1:
            others = np.delete(dist_feasible[i], i)
            min_to_others = np.min(others) if len(others) > 0 else dist_to_depot[i]
        else:
            min_to_others = dist_to_depot[i]
        min_dist_to_alternatives[i] = min(min_to_others, dist_to_depot[i])
    
    # Score: lower is better
    # We want to minimize: dist_from_current + penalty for low regret
    # Regret = min_dist_to_alternatives - dist_from_current (if positive, the node is isolated)
    
    # Actually, let's use a weighted score:
    # score[i] = w1 * dist_from_current[i] / max_dist + w2 * (1 - min_dist_to_alternatives[i] / max_alt_dist)
    # where nodes with high min_dist_to_alternatives (isolated) get lower score (preferred)
    
    # Simpler: score = dist_from_current[i] - alpha * min_dist_to_alternatives[i]
    # This favors nodes that are close to current but far from alternatives (isolated)
    
    dists_feasible = dists_from_current[feasible_indices]
    
    # Normalize for stability
    max_dist = np.max(dists_feasible) if np.max(dists_feasible) > 0 else 1.0
    max_alt = np.max(min_dist_to_alternatives) if np.max(min_dist_to_alternatives) > 0 else 1.0
    
    # Weighted score: lower is better
    alpha = 0.3  # weight for isolation/regret
    
    scores = dists_feasible / max_dist - alpha * min_dist_to_alternatives / max_alt
    
    # Add a small demand-based bonus: prefer nodes with higher demand (to exhaust capacity)
    max_demand = np.max(demands_unvisited[feasible_indices]) if np.max(demands_unvisited[feasible_indices]) > 0 else 1.0
    demand_bonus = -0.1 * demands_unvisited[feasible_indices] / max_demand
    
    scores = scores + demand_bonus
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    
    return int(unvisited_arr[feasible_indices[best_idx]])

# EVOLVE-BLOCK-END
