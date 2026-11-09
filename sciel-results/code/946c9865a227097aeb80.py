"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a combined heuristic that considers:
    1. Distance from current node
    2. Distance to the destination node (to avoid long closing edge)
    3. Isolation penalty (prefer nodes close to other unvisited nodes)
    
    Returns the index of the selected node.
    """
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])

    # Get distances from current node to all candidates
    d_current = distance_matrix[current_node, candidates]
    
    # Get distances from candidates to destination node
    d_dest = distance_matrix[candidates, destination_node]
    
    # Compute isolation score: for each candidate, find the minimum distance to other unvisited nodes
    # This is expensive, so we approximate by using the mean distance to other unvisited nodes
    # or a cheaper proxy: distance to the centroid of unvisited nodes
    
    # Cheaper approach: use the mean distance from each candidate to all other unvisited nodes
    # But this is O(n^2). For 50 cities, it's manageable but let's try a faster proxy.
    
    # Fast proxy: for each candidate, compute its distance to the centroid of unvisited nodes
    # This encourages visiting nodes that are "central" to the remaining cluster
    
    # Compute centroid of unvisited nodes (in the distance matrix space)
    # Since we only have distances, we can't compute true centroid, but we can use
    # the node with minimum mean distance to others as a proxy for "center"
    
    # Alternative: use the minimum distance from each candidate to any other unvisited node
    # This is O(n^2) but n is at most 50, so it's fine
    
    if n <= 30:
        # Compute min distance from each candidate to other unvisited nodes
        # dist_matrix[candidates][:, candidates] gives pairwise distances
        sub_matrix = distance_matrix[np.ix_(candidates, candidates)]
        # Set diagonal to inf to exclude self
        np.fill_diagonal(sub_matrix, np.inf)
        min_dist_to_others = np.min(sub_matrix, axis=1)
    else:
        # For larger n, use a cheaper approximation: mean distance to a random subset
        # or just use the distance to the destination as a proxy for isolation
        # Let's use mean distance to all other unvisited nodes
        sub_matrix = distance_matrix[np.ix_(candidates, candidates)]
        np.fill_diagonal(sub_matrix, 0)
        min_dist_to_others = np.mean(sub_matrix, axis=1)

    # Combine the scores:
    # We want to minimize: alpha * d_current + beta * d_dest + gamma * min_dist_to_others
    # The idea: 
    # - d_current: standard nearest neighbor
    # - d_dest: avoid nodes that are far from destination (to keep closing edge short)
    # - min_dist_to_others: prefer nodes that are close to other unvisited nodes (less isolated)
    
    # Normalize each component to [0, 1] range for fair combination
    d_current_norm = d_current / (np.max(d_current) + 1e-10)
    d_dest_norm = d_dest / (np.max(d_dest) + 1e-10)
    min_dist_norm = min_dist_to_others / (np.max(min_dist_to_others) + 1e-10)
    
    # Weights: tune these
    alpha = 0.5   # weight for distance from current
    beta = 0.3    # weight for distance to destination
    gamma = 0.2   # weight for isolation
    
    combined_score = alpha * d_current_norm + beta * d_dest_norm + gamma * min_dist_norm
    
    best_idx = np.argmin(combined_score)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
