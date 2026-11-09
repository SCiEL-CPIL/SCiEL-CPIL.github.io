"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def _compute_mst_leaves(nodes, distance_matrix):
    """
    Compute MST of the subgraph induced by nodes using Prim's algorithm.
    Return a boolean array indicating which nodes are leaves in the MST.
    """
    n = len(nodes)
    if n <= 1:
        return np.ones(n, dtype=bool)
    
    # Build local distance matrix
    mst_dist = distance_matrix[np.ix_(nodes, nodes)]
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    parent = np.full(n, -1, dtype=int)
    
    for _ in range(n):
        # Find unvisited node with minimum edge
        unvisited_mask = ~in_mst
        if not np.any(unvisited_mask):
            break
        candidates = np.where(unvisited_mask)[0]
        u = candidates[np.argmin(min_edge[candidates])]
        in_mst[u] = True
        
        # Update min_edge for neighbors
        for v in range(n):
            if not in_mst[v] and mst_dist[u, v] < min_edge[v]:
                min_edge[v] = mst_dist[u, v]
                parent[v] = u
    
    # Count degree in MST to identify leaves
    degree = np.zeros(n, dtype=int)
    for v in range(1, n):
        if parent[v] >= 0:
            degree[v] += 1
            degree[parent[v]] += 1
    
    # Leaves have degree <= 1
    is_leaf = degree <= 1
    return is_leaf


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using an MST-leaf pruning heuristic:
    1. Build MST of unvisited_nodes + destination_node
    2. Identify leaves of this MST (nodes with degree 1)
    3. Among the leaves, select the one with minimum distance from current_node
    4. If no leaves exist (shouldn't happen for n>=2), fall back to nearest neighbor
    
    This forces the tour to follow the global skeleton (MST) of the remaining
    subproblem, avoiding myopic local choices that lead to expensive connections.
    """
    if _check_deadline():
        # Safety fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Include destination in the MST computation
    mst_nodes = np.append(unvisited_nodes, destination_node)
    n_mst = len(mst_nodes)
    
    # Compute MST leaves
    is_leaf_all = _compute_mst_leaves(mst_nodes, distance_matrix)
    
    # The destination is the last element (index n_unvisited)
    # We only care about leaves among unvisited_nodes (indices 0 to n_unvisited-1)
    is_leaf_unvisited = is_leaf_all[:n_unvisited]
    
    if np.any(is_leaf_unvisited):
        # Among MST leaves, pick the one closest to current_node
        leaf_indices = np.where(is_leaf_unvisited)[0]
        leaf_nodes = unvisited_nodes[leaf_indices]
        leaf_dists = distance_matrix[current_node, leaf_nodes]
        best_leaf_idx = leaf_indices[np.argmin(leaf_dists)]
        return int(unvisited_nodes[best_leaf_idx])
    else:
        # Fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
