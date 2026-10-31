"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    Minimizes the sum of the distance from the current node to the candidate
    and the distance from the candidate to the destination node.
    This approximates the cost of inserting the candidate into the path
    between current and destination.
    """
    if len(unvisited_nodes) == 0:
        return -1

    # Vectorized calculation of costs
    # dist(current, candidates) + dist(candidates, destination)
    d_current_to_cand = distance_matrix[current_node, unvisited_nodes]
    d_cand_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    costs = d_current_to_cand + d_cand_to_dest
    
    min_idx = np.argmin(costs)
    
    return int(unvisited_nodes[min_idx])
# EVOLVE-BLOCK-END
