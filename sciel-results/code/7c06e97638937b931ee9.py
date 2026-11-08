"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're close to the deadline, fall back to nearest neighbor
    if time.time() - _start_time > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances from current node to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from destination to all unvisited nodes
    d_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Score: balance between proximity to current and proximity to destination
    # This helps avoid leaving the destination far away until the end
    score = d_from_current + 0.5 * d_to_dest
    
    # If only a few nodes left, use a simpler strategy
    if n_unvisited <= 5:
        # When few nodes remain, prefer going to the one closest to destination
        # to minimize the final closing cost
        return int(unvisited_nodes[np.argmin(d_to_dest)])
    
    # Select the node with minimum combined score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
