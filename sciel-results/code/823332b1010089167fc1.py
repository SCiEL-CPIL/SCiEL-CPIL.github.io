"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit using a heuristic that balances:
    1. Distance to candidate node
    2. Capacity feasibility
    3. Future cost (distance to nearest feasible unvisited or depot)
    
    Returns 0 (depot) if no feasible unvisited node exists or if returning is optimal.
    """
    if not unvisited_nodes:
        return depot
    
    # Convert to list if needed
    unvisited_list = list(unvisited_nodes)
    
    # Filter feasible nodes (demand <= rest_capacity)
    feasible = []
    for node in unvisited_list:
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if not feasible:
        return depot
    
    # If only one feasible node, go there
    if len(feasible) == 1:
        return feasible[0]
    
    # For each feasible candidate, compute a score:
    # score = dist(current, candidate) + alpha * dist(candidate, depot) + beta * dist(candidate, nearest_feasible_after)
    # We want to minimize this score.
    
    # Precompute distances from current to all feasible
    dist_from_current = np.array([distance_matrix[current_node, n] for n in feasible])
    
    # Distance from each feasible candidate to depot
    dist_to_depot = np.array([distance_matrix[n, depot] for n in feasible])
    
    # For each candidate, estimate the cost of continuing after visiting it:
    # This is approximated by the minimum distance from that candidate to any other feasible node
    # (or to depot if it's the last one). This captures the "future cost" aspect.
    
    # To be efficient, for each candidate i, we want min over j != i of dist(candidate_i, candidate_j)
    # plus consideration of whether we can still return to depot.
    
    n_feasible = len(feasible)
    if n_feasible == 1:
        return feasible[0]
    
    # Build a small distance matrix among feasible nodes
    feasible_arr = np.array(feasible)
    # dist_matrix among feasible nodes
    sub_dist = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    
    # For each candidate i, the nearest other feasible node distance
    # Set diagonal to inf so we don't pick itself
    np.fill_diagonal(sub_dist, np.inf)
    nearest_other_dist = np.min(sub_dist, axis=1)
    
    # Score: immediate distance + weight * (distance to depot + nearest other feasible)
    # The idea: prefer nodes that are close now AND don't leave us in a bad position
    # We use a weighted combination.
    
    # Normalize components to make them comparable
    # Simple approach: use raw distances with tuned weights
    # Let's try: score = dist_from_current + 0.5 * dist_to_depot + 0.3 * nearest_other_dist
    
    alpha = 0.5  # weight for distance to depot (encourages routes that can return)
    beta = 0.3   # weight for proximity to other feasible nodes (encourages clustering)
    
    scores = dist_from_current + alpha * dist_to_depot + beta * nearest_other_dist
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
