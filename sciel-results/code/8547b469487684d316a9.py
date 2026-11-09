"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0 # Safety margin before 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node to visit in a TSP tour construction.
    
    Strategy: 
    Uses a look-ahead heuristic that minimizes the sum of:
    1. Distance from current node to candidate.
    2. Distance from candidate to its nearest neighbor among the remaining unvisited nodes.
    
    This approximates a 2-opt style insertion cost, encouraging the tour to stay compact 
    and avoid creating long 'bridges' between distant clusters of unvisited nodes.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to nearest neighbor if time is running out
        if len(unvisited_nodes) == 0:
            return destination_node
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    if len(unvisited_nodes) == 0:
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Vectorized computation for speed
    # 1. Distance from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each unvisited node to all other unvisited nodes
    # dist_matrix is N x N. We need a submatrix for unvisited x unvisited.
    # unvisited_nodes is an array of indices.
    # sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # This can be expensive for large N, but N=50 is small.
    # For N=50, 50x50 matrix extraction is very fast.
    
    # To find nearest neighbor for each node in unvisited_nodes within unvisited_nodes:
    # We need min distance for each row in the submatrix.
    # Note: The distance to itself is 0, so we should ignore self or handle it.
    # Since we want nearest *other* node, we can set diagonal to inf before min, 
    # or just take min and if it's 0 (self), it doesn't matter much for ranking 
    # unless there's a tie, but usually distinct nodes have dist > 0.
    # However, strictly speaking, "nearest unvisited node" usually implies another node.
    # If we include self, the min is always 0. That would make the second term always 0.
    # The prompt says "edge from the candidate to the nearest unvisited node".
    # If it means "nearest *other* unvisited node", we must exclude self.
    
    # Let's compute the submatrix
    idx = unvisited_nodes
    sub_dist = distance_matrix[np.ix_(idx, idx)]
    
    # Set diagonal to infinity to exclude self-neighborhood
    np.fill_diagonal(sub_dist, np.inf)
    
    # Find the minimum distance for each row (each candidate)
    nearest_dist = np.min(sub_dist, axis=1)
    
    # Total cost: dist_from_current + nearest_dist
    # We want to minimize this sum.
    total_cost = dist_from_current + nearest_dist
    
    # Select the index with the minimum total cost
    best_local_idx = np.argmin(total_cost)
    
    return int(unvisited_nodes[best_local_idx])
# EVOLVE-BLOCK-END
