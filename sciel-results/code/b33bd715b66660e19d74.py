"""Improved heuristic for CVRP using regret-based selection."""
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    # Convert to arrays for speed
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    demands_arr = np.array(demands, dtype=float)
    dist_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Filter feasible nodes
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If only one feasible node, take it
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Distances from current node to all feasible nodes
    dists_current = dist_matrix[current_node, feasible_nodes]
    # Distances from depot to all feasible nodes
    dists_depot = dist_matrix[depot, feasible_nodes]
    # Demand of feasible nodes
    feas_demands = demands_arr[feasible_nodes]
    
    # Start with nearest neighbor from depot
    if current_node == depot:
        best_idx = np.argmin(dists_current)
        return int(feasible_nodes[best_idx])
    
    # Compute 2-regret score for each feasible node
    # For each node j, regret = 2nd_best_score - best_score (where best is j's own score)
    # We use a composite score: dist_to_current - alpha * dist_to_depot + beta * demand_penalty
    
    alpha = 0.7
    beta = 0.12
    
    # Base scores: lower is better
    base_scores = dists_current - alpha * dists_depot + beta * (1.0 - feas_demands / feas_demands.max())
    
    # Find the best and second-best scores
    # Sort indices by score
    sorted_indices = np.argsort(base_scores)
    best_idx = sorted_indices[0]
    second_best_idx = sorted_indices[1]
    
    # Regret for each node: how much worse it would be if we didn't pick this node
    # For node i, regret = min(score of nodes != i) - score[i]
    # But computing this for all nodes is O(n^2). Instead, we use a simpler approach:
    # The node with the smallest regret is the one where the difference between 1st and 2nd best is smallest
    # Actually, let's compute regret properly but efficiently.
    
    # For each node i, regret[i] = min(base_scores[j] for j != i) - base_scores[i]
    # This is the loss if we skip node i
    
    # Efficient computation:
    # min_excluding_i = min(base_scores[i-1], base_scores[i+1]) in sorted order
    # But we need to map back to original indices
    
    # Let's compute the regret vector
    regret = np.zeros(n_feasible)
    for i in range(n_feasible):
        # Find the minimum score excluding node i
        if i == 0:
            min_excluding = base_scores[1]
        elif i == n_feasible - 1:
            min_excluding = base_scores[n_feasible - 2]
        else:
            min_excluding = min(base_scores[i-1], base_scores[i+1])
        regret[i] = min_excluding - base_scores[i]
    
    # Select node with maximum regret (most "critical" to pick now)
    # But also consider the base score - we want a balance
    # Use: combined_score = base_score - gamma * regret
    gamma = 0.3
    combined_scores = base_scores - gamma * regret
    
    best_idx = np.argmin(combined_scores)
    
    # Check if extending the route is beneficial vs returning to depot
    dist_to_depot = dist_matrix[current_node, depot]
    best_score = base_scores[best_idx]
    
    # If the best extension is significantly worse than going back to depot, return
    # This is a soft check - we still extend if it's slightly better
    threshold = dist_to_depot * 0.95
    if best_score > threshold and n_feasible < 10:
        # Few nodes left, and extension is costly - go back
        return 0
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
