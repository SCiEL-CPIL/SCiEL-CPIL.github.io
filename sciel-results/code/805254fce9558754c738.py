"""Rewritten program for TSP constructive heuristic using minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a refined minimax strategy:
    For each candidate in unvisited_nodes, compute a score based on the maximum
    of:
    1. The distance from the current node to the candidate.
    2. The maximum distance from the candidate to any other unvisited node.
    
    Choose the candidate that minimizes this score. This prevents taking a very
    long edge from the current node and also ensures we don't pick a node that is 
    isolated from the rest of the unvisited set.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    unvisited_idx = unvisited_nodes
    
    # Get distances from current node to all unvisited nodes
    # shape: (n_unvisited,)
    dist_from_current = distance_matrix[current_node, unvisited_idx]
    
    # Get pairwise distances among unvisited nodes
    # shape: (n_unvisited, n_unvisited)
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # For each candidate i (row i), find the max distance to any OTHER unvisited node.
    # Since distances are non-negative and there is at least one other node,
    # the max of the row will be the distance to the farthest other node.
    max_dist_to_others = np.max(submatrix, axis=1)
    
    # The score for each candidate is the maximum of the step taken and the remaining bottleneck
    scores = np.maximum(dist_from_current, max_dist_to_others)
    
    # Choose the candidate with the minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
