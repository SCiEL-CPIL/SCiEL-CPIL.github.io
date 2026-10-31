"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        u1 = int(unvisited_nodes[0])
        u2 = int(unvisited_nodes[1])
        
        # Path if we pick u1: current -> u1 -> u2 -> destination
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        
        # Path if we pick u2: current -> u2 -> u1 -> destination
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 <= cost2:
            return u1
        else:
            return u2

    # General case: 1-step Lookahead
    # We want to minimize d(current, u) + d(u, v) where v is the nearest neighbor of u in unvisited
    best_node = -1
    best_score = np.inf
    
    # Pre-fetch distances from current to all unvisited for speed
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for i, u in enumerate(unvisited_nodes):
        u_int = int(u)
        
        # Get distances from u to all unvisited nodes
        dists_from_u = distance_matrix[u_int, unvisited_nodes]
        
        # Exclude u itself from being its own nearest neighbor
        # We can create a copy and set the diagonal to inf, or just handle argmin carefully
        # Since unvisited_nodes is an array of indices, and distance_matrix is a matrix,
        # dists_from_u[i] is the distance from u to u (which is 0).
        # We must ignore this.
        
        # Create a modified array for argmin
        if n_unvisited > 1:
            # Copy to avoid modifying original if we were reusing, but here it's local
            dists_modified = dists_from_u.copy()
            dists_modified[i] = np.inf
            nearest_idx = np.argmin(dists_modified)
            nearest_dist = dists_modified[nearest_idx]
        else:
            nearest_dist = 0.0
            
        score = dists_from_current[i] + nearest_dist
        
        if score < best_score:
            best_score = score
            best_node = u_int

    return best_node
# EVOLVE-BLOCK-END
