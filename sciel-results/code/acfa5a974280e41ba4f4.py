"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management if needed, though per-call overhead is low
_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin from 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Destination-Aware Greedy Heuristic.
    Instead of purely minimizing distance to the current node (Nearest Neighbor),
    it minimizes a composite cost: distance(current, candidate) + lambda * distance(candidate, destination).
    This lookahead helps avoid paths that lead to isolated clusters or long return legs.
    """
    # Check time budget to ensure we don't exceed evaluator limits
    # This check is cheap. If we are close to the limit, we fall back to pure NN or return immediately.
    if time.time() - _START_TIME > TIME_LIMIT:
        # Fallback to pure NN to be safe and fast
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Vectorized computation of the heuristic score
    # d_current = distance from current to each unvisited node
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # d_dest = distance from each unvisited node to the destination
    # Note: distance_matrix is symmetric for Euclidean TSP, but we use the index correctly.
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combine scores. Lambda = 0.5 is a standard balance between local proximity and global goal.
    # Tuning Lambda is a constant, but the *structure* (adding the term) is the change.
    lambda_val = 0.5
    scores = d_current + lambda_val * d_dest
    
    # Find the index of the minimum score in the unvisited_nodes array
    min_idx = np.argmin(scores)
    
    return int(unvisited_nodes[min_idx])
# EVOLVE-BLOCK-END
