"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE_LIMIT = 145.0  # 150s limit, 5s margin

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _check_deadline():
    _init_time()
    return (time.time() - _start_time) > _DEADLINE_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic with destination bias.
    
    Score(u) = dist(current, u) + dist(u, nearest_unvisited_neighbor) + dist(u, destination)
    
    This avoids myopic nearest-neighbor choices that lead to expensive connections
    later in the tour, especially the final return to the destination.
    """
    _init_time()
    
    # If deadline is approaching, use the fastest fallback: Nearest Neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    # Base case: only one node left
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # 1. Distance from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each unvisited to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Distance from each unvisited to its nearest neighbor in unvisited
    #    Create submatrix of distances among unvisited nodes
    #    np.ix_ creates an open mesh from indices
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so a node doesn't pick itself
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find nearest neighbor index for each row
    nn_indices = np.argmin(sub_matrix, axis=1)
    
    # Extract the distance to that nearest neighbor
    # sub_matrix[i, nn_indices[i]] gives the distance for each i
    d_nn = sub_matrix[np.arange(n_unvisited), nn_indices]
    
    # 4. Combine scores
    # Weighting: 1.0 for current, 1.0 for next, 1.0 for destination
    # This can be tuned, but 1.0 is a standard starting point for lookahead heuristics
    scores = d_curr + d_nn + d_dest
    
    # 5. Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
