# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a local density optimization heuristic.
    
    The score for each candidate node is a combination of:
    1. Distance from the current node (immediate connection cost)
    2. A 'gap' penalty that estimates how well the candidate fits into the
       local neighborhood structure, by considering its distance to the 
       destination and its nearest unvisited neighbor.
    
    This vectorized approach avoids full tour simulation and focuses on
    immediate local geometric context to make a greedy but informed choice.
    """
    if len(unvisited_nodes) == 0:
        # Fallback: shouldn't happen per contract, but safe guard
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Vectorized distance calculations
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute nearest-neighbor distance for each candidate among unvisited nodes
    # This captures local density: a node with a very close unvisited neighbor 
    # may be 'cheaper' to visit now because we can chain to it efficiently.
    # For each candidate i, find min distance from i to all other unvisited nodes.
    # We use the submatrix of distance_matrix restricted to unvisited_nodes.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row (candidate), the minimum distance to other nodes in unvisited
    # We need to exclude self-distance (which is 0), so set diagonal to inf first
    # But since we're taking min, and diagonal is 0, we need to be careful.
    # Instead, compute min distance to OTHER unvisited nodes.
    
    # Efficient way: for each candidate, the nearest neighbor among unvisited (excluding self)
    # Set diagonal to a large value before taking min
    sub_matrix_copy = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_copy, np.inf)
    nearest_unvisited_dist = np.min(sub_matrix_copy, axis=1)
    
    # Combine scores:
    # Primary: distance from current node
    # Secondary: penalty if the candidate is far from its local neighborhood 
    #   (i.e., high nearest_unvisited_dist suggests it's isolated, so maybe defer it)
    #   Actually, we want to prefer nodes that are well-connected locally.
    #   So we add a small weight to nearest_unvisited_dist as a penalty for isolation.
    # Tertiary: distance to destination - we don't want to go too far from the 
    #   destination if we're close, but this is a secondary concern.
    
    # Normalize components for stable weighting
    # Use a soft weighting scheme
    weight_current = 1.0
    weight_nearest = 0.1
    weight_dest = 0.05
    
    score = (weight_current * dist_from_current + 
             weight_nearest * nearest_unvisited_dist + 
             weight_dest * dist_to_destination)
    
    # Pick the candidate with the lowest score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
