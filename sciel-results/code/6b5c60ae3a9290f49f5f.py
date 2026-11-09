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
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Connectivity-aware heuristic with destination bias
    # Score(c) = alpha * d(current, c) + beta * d(c, dest) + gamma * (d(c, nn1) + d(c, nn2))
    # where nn1, nn2 are the two nearest neighbors of c among unvisited nodes (excluding c itself)
    # This encourages nodes that are well-connected and not too far from destination.
    
    # Pre-compute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # Extract submatrix of distances between candidates
    dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    
    # For each candidate, find the sum of distances to its two nearest neighbors
    # among the other unvisited nodes
    nn2_costs = np.zeros(n)
    
    for i in range(n):
        # Get distances from candidate i to all other candidates
        dists_i = dist_sub[i].copy()
        dists_i[i] = np.inf  # Exclude self
        
        if n >= 3:
            # Find the two smallest distances
            idx1 = np.argmin(dists_i)
            dist1 = dists_i[idx1]
            dists_i[idx1] = np.inf
            idx2 = np.argmin(dists_i)
            dist2 = dists_i[idx2]
            nn2_costs[i] = dist1 + dist2
        else:
            # n == 2: only one other node
            nn2_costs[i] = np.min(dists_i)
    
    # Weights: balance between proximity to current, destination, and connectivity
    alpha = 1.0    # weight for distance from current
    beta = 0.3     # weight for distance to destination (encourages closing the tour)
    gamma = 0.2    # weight for local connectivity (prevents isolated nodes)
    
    scores = alpha * dist_from_current + beta * dist_to_dest + gamma * nn2_costs
    
    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
