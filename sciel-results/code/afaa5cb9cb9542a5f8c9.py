"""Geometric TSP constructive heuristic: angle-deviation penalty for smooth tour construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 2 seconds before the 120s limit (safety margin)."""
    return _start_time + 118.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using geometric angle-deviation scoring.
    
    Strategy: Minimize a combination of:
    1. Distance from current node (nearest-neighbor component)
    2. Angle deviation from the straight line to destination (smoothness penalty)
    
    The angle deviation is computed using the cross-product magnitude of vectors
    (current->candidate) and (current->destination), normalized by their magnitudes.
    This enforces a smooth, non-crossing path that mimics optimal Euclidean tour curvature.
    
    Score = d(current, cand) / mean_d + beta * sin_angle_penalty
    
    where sin_angle_penalty = |cross(v1, v2)| / (|v1| * |v2|) measures how far
    the candidate deviates from the direct path to destination.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if running out of time, use nearest neighbor
    if time.time() > _get_deadline():
        dists_nn = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_nn)])
    
    # Extract coordinates implicitly through distance matrix geometry
    # We need to work with the distance matrix to compute angle deviations
    # For Euclidean TSP, we can recover relative geometry from distances
    
    # Compute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from current to destination
    d_cur_dest = distance_matrix[current_node, destination_node]
    
    # Distance from each unvisited to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Total nodes for progress calculation
    total_nodes = distance_matrix.shape[0]
    
    # Progress: 0 when starting, 1 when nearly done
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # Adaptive beta: angle penalty weight increases with progress
    # Early: more freedom to explore (small beta)
    # Late: stronger pull toward destination direction (large beta)
    beta = 0.1 + 0.9 * (progress ** 1.5)
    
    # Compute angle deviation penalty using law of cosines
    # For triangle (current, candidate, destination):
    # cos(angle_at_current) = (d_cur_cand^2 + d_cur_dest^2 - d_cand_dest^2) / (2 * d_cur_cand * d_cur_dest)
    # sin(angle) = sqrt(1 - cos^2(angle))
    
    # Handle edge case where current == destination
    if d_cur_dest < 1e-10:
        # If at destination, just pick nearest
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Avoid division by zero
    safe_dists_from = np.maximum(dists_from_current, 1e-10)
    safe_d_cur_dest = max(d_cur_dest, 1e-10)
    
    # Compute cos of angle at current node for each candidate
    # Using law of cosines on triangle (current, cand, dest)
    numerator = dists_from_current**2 + d_cur_dest**2 - dists_to_dest**2
    denominator = 2.0 * safe_dists_from * safe_d_cur_dest
    
    cos_angles = numerator / denominator
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # sin of angle: this is our deviation measure (0 = on direct path, 1 = perpendicular)
    sin_angles = np.sqrt(np.maximum(0.0, 1.0 - cos_angles**2))
    
    # Normalize distance component for scale invariance
    mean_dist = np.mean(dists_from_current)
    if mean_dist > 1e-10:
        dists_norm = dists_from_current / mean_dist
    else:
        dists_norm = dists_from_current
    
    # Combined score: distance + angle penalty
    scores = dists_norm + beta * sin_angles
    
    # Final deadline check
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
