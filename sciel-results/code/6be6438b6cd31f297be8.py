"""Constructive TSP heuristic using MST-based residual cost estimation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds, with safety margin
_TIME_MARGIN = 5.0  # seconds before hard limit


def _check_time():
    """Return True if we are within the time budget."""
    return (time.time() - _START_TIME) < (_DEADLINE - _TIME_MARGIN)


def _mst_weight(nodes, distance_matrix):
    """
    Compute the total weight of the Minimum Spanning Tree over the given nodes.
    Uses Prim's algorithm starting from the first node in the list.
    
    Args:
        nodes: array of node indices
        distance_matrix: full distance matrix
    
    Returns:
        Total MST weight
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Map node indices to positions
    node_set = set(nodes.tolist())
    
    # Prim's algorithm
    in_mst = [False] * n
    min_edge = [np.inf] * n
    min_edge[0] = 0.0
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find the vertex with minimum edge weight not yet in MST
        min_val = np.inf
        min_idx = -1
        for i in range(n):
            if not in_mst[i] and min_edge[i] < min_val:
                min_val = min_edge[i]
                min_idx = i
        
        if min_idx == -1:
            break
        
        in_mst[min_idx] = True
        total_weight += min_edge[min_idx]
        
        # Update min_edge for all other vertices
        current_node = nodes[min_idx]
        for j in range(n):
            if not in_mst[j]:
                other_node = nodes[j]
                w = distance_matrix[current_node, other_node]
                if w < min_edge[j]:
                    min_edge[j] = w
    
    return total_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour construction.
    
    Uses MST-based residual cost estimation:
    For each candidate node, estimate the remaining tour cost as:
      - Distance from current to candidate (immediate cost)
      - MST weight of the remaining unvisited nodes (residual connectivity cost)
    
    This prioritizes moves that preserve low-cost connectivity for the
    remaining sub-tour, leading to more globally optimal tour structures.
    """
    if len(unvisited_nodes) == 0:
        raise ValueError("No unvisited nodes")
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if not _check_time():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_candidates = len(unvisited_nodes)
    
    if n_candidates == 1:
        return int(unvisited_nodes[0])
    
    # For each candidate, compute:
    # 1. Immediate distance from current node
    # 2. MST weight of the remaining unvisited nodes after removing this candidate
    # 3. Combined score
    
    best_score = np.inf
    best_idx = 0
    
    # Precompute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For efficiency, compute MST for each candidate removal
    # When n is small, this is fast; when n is large, we use a faster approximation
    
    if n_candidates <= 15:
        # Exact MST computation for small sets
        for i in range(n_candidates):
            # Remove candidate i from unvisited
            remaining = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
            
            if len(remaining) == 0:
                residual_cost = 0.0
            else:
                residual_cost = _mst_weight(remaining, distance_matrix)
            
            # Total estimated tour cost from this point:
            # immediate + residual
            total_cost = dist_from_current[i] + residual_cost
            
            if total_cost < best_score:
                best_score = total_cost
                best_idx = i
    else:
        # For larger sets, use a faster heuristic:
        # Estimate residual cost as average nearest-neighbor distance among remaining
        # This is O(n^2) instead of O(n^3) for MST
        
        # Compute pairwise distances among unvisited nodes
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # For each candidate, estimate the residual cost
        for i in range(n_candidates):
            # Remove candidate i
            remaining_indices = np.concatenate([np.arange(i), np.arange(i+1, n_candidates)])
            
            if len(remaining_indices) == 0:
                residual_cost = 0.0
            elif len(remaining_indices) == 1:
                residual_cost = 0.0
            else:
                # Fast approximation: sum of minimum outgoing edge for each remaining node
                # divided by 2 (each edge counted twice)
                remaining_sub = sub_matrix[np.ix_(remaining_indices, remaining_indices)]
                # Set diagonal to inf to avoid self-loops
                np.fill_diagonal(remaining_sub, np.inf)
                min_edges = np.min(remaining_sub, axis=1)
                residual_cost = np.sum(min_edges) / 2.0
            
            total_cost = dist_from_current[i] + residual_cost
            
            if total_cost < best_score:
                best_score = total_cost
                best_idx = i
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
