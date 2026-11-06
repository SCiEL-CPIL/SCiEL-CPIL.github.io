"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Regret-based heuristic: for each feasible customer, compute the difference
    between the best and second-best insertion costs (distance from current to
    customer vs distance from current to second-best). Prioritize nodes with
    high regret (i.e., if we don't visit them now, we'll likely have to take a
    much longer detour later).
    
    Also consider the distance to depot: if the best remaining option is to go
    back to depot, we should close the route.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Get distances from current_node to all unvisited nodes
    dists_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        # No feasible customers left, go back to depot
        return depot

    # Get the feasible unvisited node indices (original node IDs)
    feasible_nodes = unvisited_nodes[feasible_indices]
    feasible_dists = dists_to_unvisited[feasible_indices]
    
    # If only one feasible node, go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute regret for each feasible node:
    # Regret = (distance to 2nd best feasible node) - (distance to this node)
    # Actually, standard regret-2: for each node i, regret = d(current, 2nd_best) - d(current, i)
    # But we want to pick the node with the highest regret, meaning the node that
    # would suffer the most if we don't visit it now.
    
    # Sort feasible distances
    sorted_dists = np.sort(feasible_dists)
    best_dist = sorted_dists[0]
    second_best_dist = sorted_dists[1] if len(sorted_dists) > 1 else best_dist
    
    # For each feasible node, compute its regret:
    # regret_i = second_best_dist - dist_i
    # The node with the largest regret is the one that is "most urgently" needed
    regrets = second_best_dist - feasible_dists
    
    # Pick the node with the highest regret
    best_idx = np.argmax(regrets)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
