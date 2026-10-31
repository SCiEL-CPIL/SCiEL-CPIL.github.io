"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    Score each candidate by: dist(current, candidate) + dist(candidate, destination_node)
    This accounts for the exit cost toward the destination, avoiding poor choices
    that are close to current but far from where we ultimately need to go.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Compute lookahead score: cost to reach candidate + cost from candidate to destination
    # This is a structural improvement over pure nearest neighbor
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score: immediate cost + exit cost toward destination
    scores = dist_to_candidates + dist_to_dest
    
    # Pick the candidate with minimum combined score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
