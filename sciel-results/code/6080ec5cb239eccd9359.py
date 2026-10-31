# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline tracking
_start_time = None
_LIMIT = 140.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Hybrid TSP constructive heuristic combining:
    1. Detour-based geometric alignment
    2. Angle deviation penalty (cross-product based) to enforce smooth paths
    3. Lightweight connectivity proxy instead of full MST
    
    The angle deviation term uses the cross-product magnitude to penalize
    candidates that deviate significantly from the straight line from current
    to destination, promoting smooth, non-crossing tour segments.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _LIMIT:
        # Fallback to nearest neighbor for safety
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 <= cost2 else c2
    
    # Compute basic distances
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # Detour score: extra distance added by going through candidate
    detours = dist_to_candidates + dist_to_dest - dist_current_to_dest
    
    # Normalize detours
    max_detour = np.max(detours)
    if max_detour > 1e-10:
        normalized_detours = detours / max_detour
    else:
        normalized_detours = detours.copy()
    
    # Angle deviation penalty using cross-product magnitude
    # For each candidate c, compute the angle between vector (current->dest) and (current->c)
    # Cross product magnitude = |v1 x v2| = |v1||v2|sin(theta)
    # Normalized: sin(theta) = |v1 x v2| / (|v1||v2|)
    # We want to penalize large angles (candidates far from the straight line)
    
    # We need coordinates to compute cross products. Since we only have distance matrix,
    # we can use the law of cosines to find cos(theta), then sin(theta).
    # cos(theta) = (|current->c|^2 + |current->dest|^2 - |c->dest|^2) / (2 * |current->c| * |current->dest|)
    # sin^2(theta) = 1 - cos^2(theta)
    # sin(theta) = sqrt(1 - cos^2(theta))
    
    # Vector from current to each candidate
    d_current_c = dist_to_candidates  # |current->c|
    d_c_dest = dist_to_dest           # |c->dest|
    d_current_dest = dist_current_to_dest  # |current->dest|
    
    # Avoid division by zero
    safe_d_current_c = np.where(d_current_c > 1e-10, d_current_c, 1e-10)
    safe_d_current_dest = max(d_current_dest, 1e-10)
    
    # Law of cosines: cos(theta) where theta is angle at current_node
    # between vectors current->candidate and current->destination
    cos_theta = (d_current_c**2 + d_current_dest**2 - d_c_dest**2) / (2.0 * safe_d_current_c * safe_d_current_dest)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # sin(theta) = sqrt(1 - cos^2(theta))
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    
    # Normalize sin_theta
    max_sin = np.max(sin_theta)
    if max_sin > 1e-10:
        normalized_angle = sin_theta / max_sin
    else:
        normalized_angle = sin_theta.copy()
    
    # Connectivity proxy: instead of expensive MST, use a simpler heuristic
    # Measure how "central" the candidate is among unvisited nodes
    # A node that is far from all other unvisited nodes may be a leaf
    # and removing it has less structural impact
    # Use average distance to other unvisited nodes as a proxy
    if n_unvisited <= 45 and (_LIMIT - elapsed) > 2.0:
        # Compute average distance from each candidate to all other unvisited nodes
        # This is O(n^2) but much cheaper than MST
        avg_dists = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            c = unvisited_nodes[i]
            # Distance from c to all other unvisited
            other_dists = distance_matrix[c, unvisited_nodes]
            # Exclude self
            other_dists[i] = 0.0
            avg_dists[i] = np.mean(other_dists)
        
        # Nodes with higher average distance are more "peripheral"
        # Removing peripheral nodes has less structural impact
        # So we want to penalize removing central nodes (low avg_dist)
        min_avg = np.min(avg_dists)
        max_avg = np.max(avg_dists)
        if max_avg > min_avg + 1e-10:
            # Normalize so that central nodes (low avg) get higher penalty
            normalized_connectivity = 1.0 - (avg_dists - min_avg) / (max_avg - min_avg)
        else:
            normalized_connectivity = np.zeros(n_unvisited)
    else:
        # Fallback: use distance to destination as proxy
        max_dest = np.max(dist_to_dest)
        if max_dest > 1e-10:
            normalized_connectivity = dist_to_dest / max_dest
        else:
            normalized_connectivity = np.zeros(n_unvisited)
    
    # Normalize local distances
    max_dist = np.max(dist_to_candidates)
    if max_dist > 1e-10:
        normalized_local = dist_to_candidates / max_dist
    else:
        normalized_local = dist_to_candidates.copy()
    
    # Adaptive weighting based on fraction of tour remaining
    fraction_remaining = n_unvisited / 50.0
    
    # Weight allocation:
    # - Local distance: always important
    # - Detour: important for path efficiency
    # - Angle deviation: important for smoothness, especially mid-tour
    # - Connectivity: important when many nodes remain
    
    alpha = 0.40 + 0.20 * (1 - fraction_remaining)  # Local distance
    beta = 0.25 + 0.15 * fraction_remaining         # Detour
    gamma = 0.15 + 0.10 * (1 - abs(fraction_remaining - 0.5) * 2)  # Angle deviation (peak mid-tour)
    delta = 0.20 * fraction_remaining                # Connectivity
    
    # Ensure weights sum to approximately 1
    weight_sum = alpha + beta + gamma + delta
    if weight_sum > 1e-10:
        alpha /= weight_sum
        beta /= weight_sum
        gamma /= weight_sum
        delta /= weight_sum
    
    scores = (alpha * normalized_local + 
              beta * normalized_detours + 
              gamma * normalized_angle + 
              delta * normalized_connectivity)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
