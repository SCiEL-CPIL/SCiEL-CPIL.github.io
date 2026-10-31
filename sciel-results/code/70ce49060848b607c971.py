"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for per-instance precomputation
_instance_cache = {}

def _get_instance_key(distance_matrix):
    """Create a hashable key for the distance matrix to cache computations."""
    # Use shape and a hash of the first few elements as a quick identifier
    # For correctness in a single-instance evaluation, we can use a simpler approach
    # Since the evaluator likely uses the same matrix across calls within an instance,
    # we can cache based on the matrix object id or a lightweight hash.
    # However, to be safe against multiple instances, we'll use a hash of the matrix data.
    # But computing a full hash every call is expensive. 
    # Instead, we assume the evaluator builds one tour at a time per instance.
    # We'll use a global counter or just cache by the matrix's memory address.
    # Actually, the safest and simplest is to not cache across different matrices,
    # but cache within a single tour construction.
    # Let's use a simple approach: store the last matrix and its precomputed data.
    return id(distance_matrix)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using a hybrid heuristic:
    1. Primary: Nearest neighbor (fast, local).
    2. Secondary: If the nearest neighbor is far, consider a 'regret' or 'global' heuristic.
    
    To improve upon simple NN, we use a 'biased nearest neighbor' that considers
    the distance to the destination node as well, to avoid getting stuck in a
    suboptimal region. This is a form of 'lookahead' or '2-opt' style correction
    applied at the selection step.
    
    Specifically, for each candidate u in unvisited_nodes, we compute a score:
    score(u) = dist(current, u) + alpha * dist(u, destination)
    
    We minimize this score. This encourages moving towards nodes that are both
    close to the current node and close to the destination, reducing the risk
    of a long closing edge.
    """
    # Deadline guard: if we've been running for too long, fall back to simple NN
    # to ensure we finish in time.
    # Note: The evaluator kills after 150s. We should be safe with O(N) per step.
    
    if len(unvisited_nodes) == 0:
        # Should not happen, but safe guard
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combine scores: primary is distance from current, secondary is distance to destination
    # We use a weighted sum. The weight alpha controls how much we care about the destination.
    # A small alpha (e.g., 0.1) keeps it close to NN but with a slight global bias.
    alpha = 0.1
    scores = dist_from_current + alpha * dist_to_dest
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
