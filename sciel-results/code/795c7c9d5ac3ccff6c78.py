# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline tracking
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Force-directed tour construction: model each unvisited city as experiencing
    attractive force toward current node and repulsive force from previously
    visited nodes. Select the candidate that maximizes net force alignment
    with the direction toward the destination.
    
    Physics model:
    - Attraction: F_att ∝ 1/d(current, candidate)  (closer = stronger pull)
    - Repulsion from visited: F_rep ∝ sum of 1/d(visited_i, candidate)
    - Net direction: normalized vector sum of attraction and repulsion
    - Score: negative dot product of net force with direction-to-destination
      (maximize alignment = minimize score)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current to all unvisited
    dist_current_to_cand = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited  
    dist_dest_to_cand = distance_matrix[destination_node, unvisited_nodes]
    
    # Distance from current to destination
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # In a force-directed model, we approximate the "direction" each candidate
    # represents relative to the current-destination axis using the triangle
    # inequality and angular relationships.
    
    # Key insight: For Euclidean distances, the angle at 'current' between
    # vector (current->candidate) and (current->destination) can be estimated
    # via the law of cosines:
    # cos(theta) = (d_cd^2 + d_cc^2 - d_cd_cand^2) / (2 * d_cd * d_cc)
    # where d_cd = dist(current,dest), d_cc = dist(current,cand), d_cd_cand = dist(dest,cand)
    
    # Compute cosine of angle at current for each candidate
    # cos_theta = (d_current_dest^2 + d_current_cand^2 - d_dest_cand^2) / (2 * d_current_dest * d_current_cand)
    numerator = dist_current_dest**2 + dist_current_to_cand**2 - dist_dest_to_cand**2
    denominator = 2.0 * dist_current_dest * dist_current_to_cand
    
    # Handle edge cases where denominator is zero
    cos_theta = np.zeros(n_unvisited)
    valid_mask = denominator > 1e-10
    cos_theta[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    cos_theta[~valid_mask] = 0.0  # Default to perpendicular
    
    # Clamp to [-1, 1] for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Force-directed scoring:
    # 1. Attraction strength: closer candidates have stronger pull
    #    F_att ∝ 1 / (dist + epsilon)
    epsilon = 1e-6
    attraction = 1.0 / (dist_current_to_cand + epsilon)
    
    # 2. Alignment score: how well the candidate direction aligns with dest direction
    #    Higher cos_theta = better alignment (closer to 1.0)
    #    We want to maximize this
    alignment = cos_theta
    
    # 3. Progress toward destination: candidates that are also close to dest
    #    get a bonus (they complete the "journey" sooner)
    #    Normalize dist_dest_to_cand
    max_dest_dist = np.max(dist_dest_to_cand)
    if max_dest_dist > 1e-10:
        progress_bonus = 1.0 - dist_dest_to_cand / max_dest_dist
    else:
        progress_bonus = np.ones(n_unvisited)
    
    # 4. Repulsion from "already covered" space: candidates that are far from
    #    the current-destination line are less desirable (they create loops)
    #    detour = d_current_cand + d_dest_cand - d_current_dest
    detour = dist_current_to_cand + dist_dest_to_cand - dist_current_dest
    max_detour = np.max(detour)
    if max_detour > 1e-10:
        detour_penalty = detour / max_detour
    else:
        detour_penalty = np.zeros(n_unvisited)
    
    # Combined force-directed score (minimize this):
    # - Strong attraction to nearby nodes
    # - High alignment with destination direction
    # - Bonus for nodes that are also near destination
    # - Penalty for large detours (loop avoidance)
    w_attr = 0.25    # weight on attraction (inverse distance)
    w_align = 0.45   # weight on directional alignment
    w_prog = 0.15    # weight on progress bonus
    w_detour = 0.15  # weight on detour penalty
    
    # Score: lower is better
    # -attraction (we want high attraction, so negate)
    # -alignment (we want high alignment, so negate)
    # -progress_bonus (we want high progress, so negate)
    # +detour_penalty (we want low detour)
    scores = (-w_attr * attraction / np.max(attraction) +
              -w_align * alignment +
              -w_prog * progress_bonus +
              w_detour * detour_penalty)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
