"""Traveling salesman constructive heuristic with improved node selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for time tracking
_START_TIME = None
_TIME_LIMIT = 110.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using a combined heuristic that considers:
    1. Distance from current node to candidate
    2. Distance from candidate to destination node
    3. A small penalty for nodes that would create long edges later
    
    This is a greedy 2-opt-like lookahead heuristic.
    """
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination node to all unvisited nodes
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Combined score: primarily distance from current, with some consideration for
    # how close the candidate is to the destination (to avoid bad final edges)
    # We use a weighted combination that emphasizes current distance but
    # slightly penalizes candidates that are far from destination
    
    # Weight: 80% current distance, 20% destination distance
    # This encourages picking nodes that are close to current but also
    # somewhat close to where we need to end up
    combined_score = 0.8 * dist_from_current + 0.2 * dist_to_dest
    
    # Additionally, we can add a small penalty based on the second-best distance
    # from the candidate to other unvisited nodes (to avoid isolated points)
    # But this is expensive, so only do it for larger instances with time to spare
    
    if n_unvisited > 5 and not _time_exceeded():
        # For each candidate, estimate the cost of the edge from candidate to
        # its nearest unvisited neighbor (excluding itself and current)
        # This is a 2-step lookahead
        # To keep it fast, we'll just use the minimum distance from candidate
        # to any other unvisited node as a proxy for connectivity
        
        # Get the submatrix of distances among unvisited nodes
        # This could be expensive for large n, so limit it
        if n_unvisited <= 30:
            unvisited_indices = unvisited_nodes
            submatrix = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
            # For each candidate i, find min distance to other unvisited nodes
            # Set diagonal to inf to exclude self
            np.fill_diagonal(submatrix, np.inf)
            min_neighbor_dist = np.min(submatrix, axis=1)
            
            # Add a small penalty for candidates with poor connectivity
            # Scale the penalty to be small relative to the main distance terms
            penalty = 0.1 * min_neighbor_dist
            combined_score = combined_score + penalty
    
    # Pick the node with the lowest combined score
    best_idx = np.argmin(combined_score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
