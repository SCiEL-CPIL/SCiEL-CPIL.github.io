"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for depot coordinates derived from distance matrix
_depot_coords_cache = {}


def _get_depot_coords(depot, distance_matrix):
    """Get depot coordinates from distance matrix using first two dimensions."""
    key = id(distance_matrix)
    if key in _depot_coords_cache:
        return _depot_coords_cache[key]
    
    # Extract coordinates from distance matrix
    # Assume Euclidean distance matrix
    # d[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
    # We can estimate coordinates using the first few rows
    n = len(distance_matrix)
    if n < 3:
        return None
    
    # Use the first three non-depot nodes to estimate 2D coordinates
    # Actually, we can use the fact that for Euclidean distances:
    # d(depot, i)^2 + d(depot, j)^2 - d(i, j)^2 = 2 * dot(vi, vj)
    # where vi = pos(i) - pos(depot)
    
    # Get distances from depot
    depot_dist = distance_matrix[depot, :]
    
    # Find first 3 nodes that are not depot
    non_depot = [i for i in range(n) if i != depot][:3]
    
    if len(non_depot) < 3:
        return None
    
    # We'll use node 1, 2, 3 (excluding depot) to set up coordinate system
    # Let's place depot at origin, and use the distances to define coordinates
    
    # For simplicity, let's just use the angle between vectors from depot
    # We need to compute angles relative to the current direction
    
    # Actually, let me use a different approach: compute the angle of each node
    # relative to the depot using the distance matrix
    
    # Store the distance matrix reference
    _depot_coords_cache[key] = distance_matrix
    return distance_matrix


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select next node using sweep-angle priority heuristic.
    Prioritizes customers that maintain consistent angular sector relative to depot.
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited)
    
    # If no unvisited nodes, close the route
    if n_unvisited == 0:
        return 0
    
    # Filter by capacity feasibility
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_indices = unvisited[feasible_mask]
    
    if len(feasible_indices) == 0:
        return 0
    
    # Compute distances from current node to feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_indices]
    
    # Compute distances from depot to feasible nodes
    dist_from_depot = distance_matrix[depot, feasible_indices]
    
    # Compute the angle of each feasible node relative to the depot
    # and the current direction (from depot to current_node)
    
    # Get distances for angle computation
    d_depot_current = distance_matrix[depot, current_node]
    
    # For each feasible node j, compute the angle between:
    # - vector from depot to current_node
    # - vector from depot to node j
    # Using the law of cosines:
    # cos(theta) = (d(depot,current)^2 + d(depot,j)^2 - d(current,j)^2) / (2 * d(depot,current) * d(depot,j))
    
    if d_depot_current < 1e-10:
        # Current node is at depot, use nearest feasible
        return int(feasible_indices[np.argmin(dist_from_current)])
    
    # Compute cosines of angles
    denom = 2 * d_depot_current * dist_from_depot
    valid_denom = denom > 1e-10
    cos_angles = np.zeros(len(feasible_indices))
    cos_angles[valid_denom] = (d_depot_current**2 + dist_from_depot[valid_denom]**2 - dist_from_current[valid_denom]**2) / denom[valid_denom]
    
    # Clamp to [-1, 1]
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # Compute angles (in radians)
    angles = np.arccos(cos_angles)
    
    # Score: prefer smaller angles (closer to current direction) and shorter distances
    # Use a combination: score = angle + alpha * normalized_distance
    # Normalize distance to [0, 1]
    min_dist = np.min(dist_from_current)
    max_dist = np.max(dist_from_current)
    if max_dist - min_dist > 1e-10:
        norm_dist = (dist_from_current - min_dist) / (max_dist - min_dist)
    else:
        norm_dist = np.zeros_like(dist_from_current)
    
    # Weighted score: smaller is better
    angle_weight = 1.0
    dist_weight = 0.3
    scores = angle_weight * angles + dist_weight * norm_dist
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_indices[best_idx])
# EVOLVE-BLOCK-END
