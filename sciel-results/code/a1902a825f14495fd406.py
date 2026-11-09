"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 15.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a composite heuristic that balances:
    1. Distance from current node to candidate (immediate cost)
    2. Distance from candidate to destination (lookahead/continuation cost)
    3. A penalty for candidates that are isolated or create long future gaps
    
    This uses a weighted sum of immediate and lookahead distances,
    which is a simple 2-opt lookahead heuristic.
    """
    # Safety check: if no unvisited nodes, return -1 (shouldn't happen in valid TSP)
    if len(unvisited_nodes) == 0:
        return -1
    
    # Check deadline - if approaching, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from each unvisited to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Composite score: immediate + lookahead
    # Weights: emphasize immediate cost slightly more, but include lookahead
    # This is a 2-step lookahead heuristic
    alpha = 1.0  # weight for immediate distance
    beta = 0.5   # weight for lookahead distance
    
    scores = alpha * dist_from_current + beta * dist_to_dest
    
    # Select the candidate with minimum composite score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
