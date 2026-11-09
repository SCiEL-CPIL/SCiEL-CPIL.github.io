"""Combined constructive TSP heuristic: geometric direction-following with convex-hull bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

# Module-level state for deadline guarding
_T0 = None
_DEADLINE = 145.0  # Safety margin before 150s kill
_SAFETY = 5.0

# Cache for convex hull membership per instance
_hull_cache = {}
_instance_id = 0

def _get_elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _compute_hull_membership(distance_matrix, unvisited_nodes):
    """
    Compute convex hull membership for the unvisited nodes.
    We extract coordinates from the distance matrix if possible, 
    or use a proxy based on distances.
    
    Since we only have a distance matrix, we need to reconstruct coordinates 
    or use a different proxy. For Euclidean instances, we can use 
    multidimensional scaling or simply use the distance to a reference point.
    
    Actually, for a distance matrix, we can't easily get coordinates. 
    Let's use a different approach: compute the "hull-ness" using 
    the number of nodes that are closer to the current node than the candidate.
    
    A node is on the convex hull if there are few nodes "inside" it.
    We can approximate this by counting how many other unvisited nodes 
    are closer to the current node than the candidate is.
    """
    global _hull_cache, _instance_id
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited < 3:
        return np.ones(n_unvisited)
    
    # Use a simple proxy: for each node, count how many other nodes are 
    # "behind" it relative to the current node. This is a rough hull approximation.
    # A better proxy: use the fact that hull nodes have large angles between 
    # their neighbors when viewed from the centroid.
    
    # For now, use a simpler heuristic: nodes with fewer close neighbors 
    # in the distance matrix are more likely on the hull.
    D_sub = distance_matrix[unvisited_nodes, unvisited_nodes]
    np.fill_diagonal(D_sub, np.inf)
    
    # For each node, find the 3 nearest neighbors among unvisited
    # Hull nodes tend to have larger distances to their nearest neighbors
    # compared to interior nodes.
    if n_unvisited > 3:
        k = min(3, n_unvisited - 1)
        # Get the k-th smallest distance for each node
        sorted_dists = np.sort(D_sub, axis=1)
        kth_dists = sorted_dists[:, k-1]
        
        # Normalize to [0, 1]
        kth_min = np.min(kth_dists)
        kth_max = np.max(kth_dists)
        if kth_max > kth_min + 1e-10:
            hull_scores = (kth_dists - kth_min) / (kth_max - kth_min)
        else:
            hull_scores = np.ones(n_unvisited) / n_unvisited
    else:
        hull_scores = np.ones(n_unvisited) / n_unvisited
    
    return hull_scores

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    
    Uses a geometric direction-following heuristic:
    - Primary signal: angle deviation from the current-to-destination vector
    - Secondary signal: convex hull bias (prefer boundary nodes)
    - Tertiary signal: nearest neighbor distance (tie-breaker)
    
    This constructs the tour by tracing the convex hull first, then filling
    interior nodes, which reduces edge crossings.
    """
    unvisited_nodes = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Hard deadline guard
    elapsed = _get_elapsed()
    if elapsed > _DEADLINE - _SAFETY:
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
    
    try:
        # Distances from current to all unvisited
        d_cur = distance_matrix[current_node, unvisited_nodes]
        
        # Distances from destination to all unvisited
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Compute geometric direction following
        # We need to estimate the "direction" from current to each candidate
        # and compare it to the direction from current to destination.
        
        # Since we only have distances, we use a proxy: 
        # The angle between vectors (current->candidate) and (current->destination)
        # can be estimated using the law of cosines:
        # cos(theta) = (d_cur^2 + d_dest^2 - d_cand_dest^2) / (2 * d_cur * d_dest)
        
        # d_cand_dest is the distance from candidate to destination
        d_cand_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Avoid division by zero
        d_cur_safe = np.maximum(d_cur, 1e-10)
        d_dest_safe = np.maximum(d_dest, 1e-10)
        
        # Compute cos(theta) using law of cosines
        # theta is the angle at current_node between (current->candidate) and (current->destination)
        numerator = d_cur_safe**2 + d_dest_safe**2 - d_cand_dest**2
        denominator = 2.0 * d_cur_safe * d_dest_safe
        cos_theta = numerator / denominator
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Angle deviation: 0 means perfectly aligned, pi means opposite
        # We want small angles (aligned with destination direction)
        # Use (1 - cos_theta) as a proxy for angle deviation
        angle_deviation = 1.0 - cos_theta
        
        # Compute convex hull bias
        # Hull nodes should be preferred early in the tour
        hull_scores = _compute_hull_membership(distance_matrix, unvisited_nodes)
        
        # Normalize scores to comparable scales
        # angle_deviation is in [0, 2]
        # d_cur is in distance units
        # hull_scores is in [0, 1]
        
        # Scale factors to balance the terms
        # We want angle_deviation to dominate, with hull as a secondary signal
        # and d_cur as a tertiary signal
        
        # Normalize d_cur to [0, 1]
        d_cur_min = np.min(d_cur)
        d_cur_max = np.max(d_cur)
        if d_cur_max > d_cur_min + 1e-10:
            d_cur_norm = (d_cur - d_cur_min) / (d_cur_max - d_cur_min)
        else:
            d_cur_norm = np.zeros(n_unvisited)
        
        # Weighted combination:
        # - angle_deviation: primary (weight 1.0)
        # - (1 - hull_scores): secondary (weight 0.3, prefer hull nodes)
        # - d_cur_norm: tertiary (weight 0.1, prefer closer nodes)
        
        score = 1.0 * angle_deviation + 0.3 * (1.0 - hull_scores) + 0.1 * d_cur_norm
        
        best_idx = int(np.argmin(score))
        return int(unvisited_nodes[best_idx])
        
    except Exception:
        # Fallback to simple NN
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
# EVOLVE-BLOCK-END
