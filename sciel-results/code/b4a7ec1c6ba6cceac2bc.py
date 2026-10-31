"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Second-Nearest-Neighbor (Lookahead) heuristic.
    
    This approach differs from simple Nearest Neighbor by considering the cost
    of the *next* move. For each candidate node, we calculate the cost to reach
    it from the current node, plus a penalty proportional to the minimum distance
    from that candidate to any other unvisited node. This avoids picking nodes
    that are close to the current position but isolated from the rest of the
    unvisited set, which would result in a long jump in the next step.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        # Exact evaluation for the last two nodes
        # Path: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Path: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    # Extract the sub-matrix for unvisited nodes
    dist_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Distance from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate i, find the minimum distance to any *other* unvisited node j
    # Set diagonal to infinity before finding min
    dist_sub_copy = dist_sub.copy()
    np.fill_diagonal(dist_sub_copy, np.inf)
    
    # min_dist_to_other[i] is the distance from unvisited_nodes[i] to its nearest neighbor in unvisited_nodes
    min_dist_to_other = np.min(dist_sub_copy, axis=1)
    
    # Heuristic Score:
    # Primary: Distance from current
    # Secondary: Penalty for being isolated (distance to next nearest)
    # We use a coefficient of 0.5 to balance the immediate cost and the future cost.
    
    scores = dist_from_current + 0.5 * min_dist_to_other
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
