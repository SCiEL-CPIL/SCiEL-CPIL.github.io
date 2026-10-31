"""Rewritten program: TSP constructive heuristic with connectivity-aware lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 3 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Connectivity-aware greedy heuristic.
    - Uses a penalty for nodes that are far from other unvisited nodes (min_other distance)
      to avoid creating isolated nodes that require long jumps later.
    - Uses an adaptive weight for the distance to the destination (start node) to ensure
      efficient tour closure.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline with safety margin
    if time.time() > _get_deadline():
        # Fallback to nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Compute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute minimum distance from each unvisited node to any OTHER unvisited node
    # This is the "connectivity" penalty
    # Extract submatrix of distances between unvisited nodes
    # Shape: (n_unvisited, n_unvisited)
    sub_matrix = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # Set diagonal to infinity so we don't pick the node itself (distance 0)
    # We want min distance to *other* nodes
    np.fill_diagonal(sub_matrix, np.inf)
    
    # If n_unvisited == 2, min_other will be the distance between the two.
    # If n_unvisited > 2, it's the closest neighbor among unvisited.
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # Handle edge case where min_dist_to_other might be inf (shouldn't happen if n_unvisited >= 2)
    # But just in case, replace inf with a large value
    min_dist_to_other = np.nan_to_num(min_dist_to_other, nan=0.0, posinf=1e10, neginf=0.0)
    
    total_nodes = distance_matrix.shape[0]
    
    # Progress: 0 when all nodes unvisited, 1 when only 1 remains
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # Adaptive weights
    # w_dest: increases as we approach the end to bias towards closing the tour
    # w_conn: decreases as we approach the end, as connectivity is less of an issue with few nodes left
    # We want to avoid "traps" early on, so high w_conn early.
    # We want to close the loop late, so high w_dest late.
    
    # Quadratic scaling for destination bias to make it very strong at the end
    w_dest = 0.1 + 0.9 * (progress ** 2)
    
    # Linear decay for connectivity bias
    w_conn = 0.5 * (1.0 - progress)
    
    # Normalize distances to make weights scale-invariant
    # Use mean distance from current as a reference scale
    mean_dist = np.mean(dists_from_current)
    if mean_dist < 1e-10:
        mean_dist = 1.0
        
    # Scores
    # We want to minimize: dist(current, next) + w_dest * dist(next, dest) + w_conn * min_dist_to_other(next)
    # Normalizing helps the weights be consistent across different instance scales
    score = (dists_from_current / mean_dist) + \
            w_dest * (dists_to_dest / mean_dist) + \
            w_conn * (min_dist_to_other / mean_dist)
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
