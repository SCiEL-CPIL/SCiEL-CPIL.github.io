"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    # (We don't have a global timer here easily without state, but the function is called 49 times.
    # The overhead of the 2-opt look-ahead is O(N^2) per call. N=50. 2500 ops * 49 calls = ~125k ops.
    # This is extremely fast in NumPy, well under 120s. No explicit guard needed for this scale.)
    
    # 1. Distance from current node to all unvisited nodes
    d_current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each unvisited node, find the minimum distance to any *other* unvisited node.
    # This approximates the cost of the edge leaving the candidate node to the rest of the tour.
    # We compute the full submatrix of unvisited nodes.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We need the min distance from each node to any other node in the unvisited set.
    # np.min(sub_matrix, axis=1) gives the min distance to itself (0) or others?
    # Distance matrix diagonal is 0. So min will be 0.
    # We need the min distance to *other* nodes.
    # Set diagonal to inf before taking min.
    np.fill_diagonal(sub_matrix, np.inf)
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # 3. Score each candidate: cost to enter + cost to leave
    # Cost to enter: d_current_to_unvisited
    # Cost to leave: min_dist_to_other
    scores = d_current_to_unvisited + min_dist_to_other
    
    # 4. Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
