"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    # If only one node left, must pick it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If two nodes left, we can calculate the exact cost of both permutations
    # Path: current -> A -> B -> destination
    # Path: current -> B -> A -> destination
    if n_unvisited == 2:
        u1 = int(unvisited_nodes[0])
        u2 = int(unvisited_nodes[1])
        
        # Cost of choosing u1 first
        cost_u1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        # Cost of choosing u2 first
        cost_u2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost_u1 <= cost_u2:
            return u1
        else:
            return u2
            
    # For > 2 nodes, use 1-step lookahead heuristic
    # Score(j) = dist(current, j) + min_{k in unvisited, k != j} dist(j, k)
    
    # Vectorized calculation for speed
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # We need min distance from each candidate j to any other unvisited k (k != j)
    # Create a mask to exclude self-distance
    # dist_matrix_unvisited is a submatrix of distance_matrix for unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find min distance for each candidate to its nearest other unvisited node
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # Total score
    scores = dist_from_current + min_dist_to_other
    
    # Find index of minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
