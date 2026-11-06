"""Traveling salesman constructive heuristic: improved nearest-neighbor with look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_TIME_BUDGET = 115.0  # seconds, safety margin before 120s limit


def _check_deadline():
    """Return True if we've exceeded the time budget."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_BUDGET


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour using an improved heuristic.
    
    Uses a composite score combining:
    1. Distance from current node (greedy component)
    2. Distance to destination (to avoid getting stuck far from destination)
    3. Local density / integration (prefer nodes well-connected to other unvisited)
    4. Look-ahead: estimate of cost if we pick this node (distance to its nearest unvisited neighbor)
    
    This aims to reduce long detours and produce shorter closed tours.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    # Edge case: only one unvisited node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if we're running out of time, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited
    dist_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    # Normalize distances to [0, 1] range for weighting
    # Use range normalization to make weights comparable
    min_cur = np.min(dist_from_current)
    max_cur = np.max(dist_from_current)
    range_cur = max_cur - min_cur if max_cur > min_cur else 1.0
    norm_cur = (dist_from_current - min_cur) / range_cur
    
    min_dest = np.min(dist_to_destination)
    max_dest = np.max(dist_to_destination)
    range_dest = max_dest - min_dest if max_dest > min_dest else 1.0
    norm_dest = (dist_to_destination - min_dest) / range_dest
    
    # Compute local density: for each unvisited node, how close is it to other unvisited nodes?
    # Lower average distance to other unvisited = better integrated (more central)
    # We use the minimum distance to another unvisited node as a proxy for local density
    # (a node with a very close neighbor is well-integrated)
    
    # For efficiency, compute distance from each unvisited to all other unvisited
    # This is O(n^2) where n = number of unvisited
    if n_unvisited <= 100:
        # Compute pairwise distances among unvisited nodes
        unvisited_indices = unvisited_nodes
        sub_matrix = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
        
        # For each node, find its nearest neighbor among other unvisited nodes
        # Set diagonal to inf to exclude self
        np.fill_diagonal(sub_matrix, np.inf)
        nearest_unvisited_dist = np.min(sub_matrix, axis=1)
        
        # Normalize
        min_nd = np.min(nearest_unvisited_dist)
        max_nd = np.max(nearest_unvisited_dist)
        range_nd = max_nd - min_nd if max_nd > min_nd else 1.0
        norm_density = (nearest_unvisited_dist - min_nd) / range_nd
    else:
        # For large n, use a simpler approximation: just use distance to current as proxy
        norm_density = np.zeros(n_unvisited)
    
    # Look-ahead: if we pick node i, what's the estimated additional cost?
    # Approximate by: distance from i to its nearest unvisited neighbor (excluding itself)
    # This is already captured in norm_density above
    
    # Composite score: lower is better
    # Weight the components to balance greedy progress, destination awareness, and local integration
    w_current = 0.55    # Weight for distance from current (primary greedy signal)
    w_dest = 0.25       # Weight for distance to destination (avoid getting stuck)
    w_density = 0.20    # Weight for local density (prefer well-integrated nodes)
    
    score = w_current * norm_cur + w_dest * norm_dest + w_density * norm_density
    
    # Select the node with the lowest composite score
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
