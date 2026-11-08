"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a heuristic that balances:
    1. Distance from current node to candidate (nearest neighbor)
    2. Distance from candidate to destination (to help close the tour)
    3. Lookahead: distance from candidate's best unvisited neighbor to destination
    
    This is a 2-step lookahead heuristic that should outperform pure nearest neighbor.
    """
    global _start_time, _DEADLINE
    
    # Time limit guard - fall back to nearest neighbor if close to deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    unvisited = unvisited_nodes
    n_unvisited = len(unvisited)
    
    if n_unvisited == 0:
        # Should not happen, but return destination as fallback
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited[0])
    
    # Get distances from current to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited]
    
    # Get distances from all unvisited to destination
    d_to_dest = distance_matrix[unvisited, destination_node]
    
    # For each candidate, find the minimum distance from that candidate to any other unvisited node
    # This is a 2-step lookahead: if we go to candidate i, what's the best next move?
    # We approximate by looking at min distance from candidate to other unvisited nodes
    
    # Efficient computation: for each candidate, min distance to other unvisited
    # distance_matrix[unvisited, unvisited] gives us the full submatrix
    submatrix = distance_matrix[np.ix_(unvisited, unvisited)]
    
    # For each row (candidate), the min distance to other unvisited (excluding self)
    # Set diagonal to inf to exclude self
    np.fill_diagonal(submatrix, np.inf)
    min_next_dist = np.min(submatrix, axis=1)
    
    # Score: balance between distance from current, distance to destination, and lookahead
    # We want to minimize a weighted combination:
    # - Primary: distance from current (don't make huge jumps)
    # - Secondary: distance to destination (helps close tour)
    # - Tertiary: min distance to next unvisited (smooth tour)
    
    # Normalize components to comparable scales
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-10)
    d_to_dest_norm = d_to_dest / (np.max(d_to_dest) + 1e-10)
    min_next_norm = min_next_dist / (np.max(min_next_dist) + 1e-10)
    
    # Weighted score - lower is better
    # Emphasize proximity to current, but also consider destination and smoothness
    score = (0.5 * d_from_current_norm + 
             0.3 * d_to_dest_norm + 
             0.2 * min_next_norm)
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(score)
    
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
