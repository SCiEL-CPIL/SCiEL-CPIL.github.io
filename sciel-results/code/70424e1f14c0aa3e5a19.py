"""Improved constructive TSP heuristic: multi-criteria greedy with adaptive weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a multi-criteria greedy heuristic.
    
    Scoring combines:
    1. Forward distance (current -> candidate)
    2. Return distance (candidate -> destination)
    3. A 'squeeze' term that penalizes candidates that would create long edges
       in the eventual tour by looking at the second-best return option
    
    The blend is adaptive: early in the tour we weight forward more,
    later we weight return more to close the loop.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: nearest neighbor
        unv = np.asarray(unvisited_nodes)
        if unv.size == 0:
            return int(destination_node)
        if unv.size == 1:
            return int(unv[0])
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    dm = distance_matrix
    
    # Forward cost: distance from current to each candidate
    d_forward = dm[current_node, unv]
    # Return cost: distance from each candidate to destination
    d_return = dm[unv, destination_node]
    
    # Adaptive weight: as fewer nodes remain, weight the return more heavily
    # Total nodes in the tour is roughly the original size; we don't know it exactly,
    # but we can use n as a proxy. When n is large, we're early; when n is small, we're late.
    # Normalize: weight_return goes from ~0.3 (early) to ~1.0 (late)
    # Use a smooth transition
    if n > 10:
        w_return = 0.3 + 0.7 * (1.0 - n / 50.0) if n <= 50 else 0.3
    else:
        w_return = 0.3 + 0.7 * (1.0 - n / 10.0)
    w_return = np.clip(w_return, 0.3, 1.0)
    
    # Squeeze term: for each candidate, what's the minimum return distance among OTHER candidates?
    # This penalizes candidates that are "isolated" in terms of return options
    # Actually, let's use a simpler but effective term:
    # Penalize if the candidate is far from the destination relative to others
    # This is already captured by d_return. Let's add a different angle:
    
    # Alternative: use the average distance from candidate to unvisited nodes as a "hub" penalty
    # But that's O(n^2) which might be slow. For 50 cities, it's fine.
    # Actually, let's try a simpler improvement:
    
    # Combine with a small penalty for candidates that are far from the centroid of unvisited nodes
    # This encourages visiting "central" nodes first to avoid long detours
    if n > 2:
        # Compute centroid of unvisited nodes (using distance matrix is not ideal, 
        # but we can use the average row as a proxy for "centrality")
        # Actually, let's just use the mean distance from each candidate to all other unvisited nodes
        # as a measure of how "central" the candidate is
        # This is O(n^2) but n <= 50 so it's fine
        d_to_others = np.mean(dm[unv[:, None], unv[None, :]], axis=1)
        # Normalize d_to_others to [0, 1]
        d_min = np.min(d_to_others)
        d_max = np.max(d_to_others)
        if d_max > d_min:
            d_norm = (d_to_others - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros(n)
        
        # Blend: forward + w_return * return + w_central * centrality_penalty
        # Centrality penalty: prefer nodes with lower average distance to others (more central)
        w_central = 0.1
        score = d_forward + w_return * d_return + w_central * d_norm
    else:
        score = d_forward + w_return * d_return
    
    return int(unv[np.argmin(score)])
# EVOLVE-BLOCK-END
