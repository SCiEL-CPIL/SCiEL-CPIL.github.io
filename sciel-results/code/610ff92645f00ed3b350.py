"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (120s hard limit, use 115s as safety margin)
_DEADLINE = 115.0
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a TSP tour being built step-by-step.
    
    Uses a look-ahead heuristic: for each candidate next node, estimate the
    cost of going current -> candidate -> (best unvisited after candidate) 
    plus the cost of eventually returning to destination. This is a 2-step
    look-ahead that captures more global structure than pure nearest neighbor.
    
    Parameters:
        current_node: int, the node we're currently at
        destination_node: int, the node we must eventually return to
        unvisited_nodes: np.ndarray, array of unvisited node indices
        distance_matrix: np.ndarray, NxN distance matrix
    
    Returns:
        int: the next node to visit
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if we're running low on time, fall back to nearest neighbor
    if _elapsed() > _DEADLINE - 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # 2-step look-ahead heuristic:
    # For each candidate i in unvisited_nodes, compute:
    #   score_i = dist(current, i) + min over j != i in unvisited of dist(i, j)
    # This captures the cost of moving to i and then to the best next stop.
    # We also add a small penalty for distance to destination to bias toward
    # nodes that keep us closer to where we need to go.
    
    # Vectorized computation for efficiency
    # dist_current_to_candidates: shape (n_unvisited,)
    dist_current = distance_matrix[current_node, unvisited_nodes]  # (n,)
    
    # For each candidate i, find min distance to any other unvisited node j != i
    # dist_matrix among unvisited: shape (n, n)
    dist_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
    
    # Set diagonal to inf so we don't pick the same node
    np.fill_diagonal(dist_unvisited, np.inf)
    
    # min_next[i] = min over j != i of dist(i, j)
    min_next = np.min(dist_unvisited, axis=1)  # (n,)
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]  # (n,)
    
    # Combined score: primary is 2-step look-ahead, secondary is distance to destination
    # Weight the destination distance lightly to avoid getting stuck far from dest
    score = dist_current + min_next + 0.1 * dist_to_dest
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
