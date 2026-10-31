"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    This approach evaluates candidates based on the cost of the path segment formed
    by choosing the candidate and then the best possible subsequent node.
    
    Score(u) = dist(current, u) + min_{v != u} (dist(u, v) + min_{w != v} dist(v, w)) + lambda * dist(u, destination)
    
    This differs from simple feature-weighted heuristics by considering the dynamic
    path structure ahead of the current step.
    """
    k = len(unvisited_nodes)
    
    if k == 0:
        return int(destination_node)
    if k == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Matrix of distances between unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Calculate the minimum distance from each node to any other unvisited node
    # This approximates the cost of the step AFTER the candidate node
    min_next_matrix = submatrix.copy()
    np.fill_diagonal(min_next_matrix, np.inf)
    min_next_costs = np.min(min_next_matrix, axis=1)
    
    # Calculate the cost of the best 2-step path starting from each candidate u
    # We want to minimize: dist(u, v) + dist(v, w)
    # We approximate dist(v, w) by min_next_costs[v] (the best next step from v)
    # So for each u, we find v that minimizes dist(u, v) + min_next_costs[v]
    
    # Construct a matrix where entry (i, j) is dist(u_i, u_j) + min_next_costs[u_j]
    # min_next_costs is added to each column j
    lookahead_costs = submatrix + min_next_costs[None, :]
    
    # We cannot choose v = u (i.e., we must move to a different node)
    np.fill_diagonal(lookahead_costs, np.inf)
    
    # The best 2-step cost starting from u_i is the minimum over all v
    best_2step_cost = np.min(lookahead_costs, axis=1)
    
    # Combine immediate cost, lookahead cost, and destination bias
    # lambda is tuned to balance the closing cost
    lambda_dest = 0.5
    scores = dist_from_current + best_2step_cost + lambda_dest * dist_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
