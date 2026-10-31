"""Rewritten program for TSP constructive heuristic using minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a minimax strategy:
    Minimizes the maximum edge length introduced by the current step and the
    potential future steps from the chosen node to the remaining unvisited nodes.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    unvisited_idx = unvisited_nodes
    n_unvisited = len(unvisited_idx)
    
    # 1. Distance from current node to all unvisited nodes
    d_current = distance_matrix[current_node, unvisited_idx]
    
    # 2. Bottleneck cost: Max distance from each unvisited node to other unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)].copy()
    # Set diagonal to -inf so max ignores self-distance
    np.fill_diagonal(submatrix, -np.inf)
    bottleneck_costs = np.max(submatrix, axis=1)
    
    # 3. Minimax Score:
    # We want to choose the next node j such that the maximum of 
    # (distance from current to j) and (max distance from j to any other unvisited node)
    # is minimized.
    scores = np.maximum(d_current, bottleneck_costs)
    
    # 4. Penalize visiting the destination node early if it's in unvisited
    # In a closed tour, the destination is the start node. We should only 
    # visit it when it is the last unvisited node to close the tour.
    if destination_node in unvisited_idx:
        dest_mask = (unvisited_idx == destination_node)
        if np.sum(dest_mask) > 0:
            # If there are other nodes to visit, strongly penalize the destination
            # to ensure it is picked last.
            if n_unvisited > 1:
                scores[dest_mask] += 1e6
    
    # 5. Select the node with the minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
