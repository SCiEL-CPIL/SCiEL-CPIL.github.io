"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 15.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def _compute_mst_leaves(unvisited_nodes, destination_node, distance_matrix):
    """
    Compute MST leaves among unvisited nodes plus destination.
    Returns a boolean mask over unvisited_nodes indicating which are leaves
    in the MST of (unvisited_nodes + destination_node).
    
    A leaf in the MST is a node with degree 1 in the tree. The idea is that
    the optimal tour must connect all remaining nodes, and the MST captures
    the global connectivity structure. By preferring to visit MST leaves
    early, we force the construction to follow the global skeleton.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited <= 1:
        return np.ones(n_unvisited, dtype=bool)
    
    # Build the subgraph: unvisited nodes + destination
    # Nodes are indexed 0..n_unvisited-1 for unvisited, n_unvisited for destination
    n_total = n_unvisited + 1
    
    # Build adjacency matrix for this subgraph
    # dist[i,j] for i,j in 0..n_unvisited-1 (unvisited to unvisited)
    # dist[i, n_unvisited] for i in 0..n_unvisited-1 (unvisited to dest)
    # dist[n_unvisited, i] = dist[i, n_unvisited]
    
    sub_dist = np.zeros((n_total, n_total), dtype=float)
    sub_dist[:n_unvisited, :n_unvisited] = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    sub_dist[:n_unvisited, n_unvisited] = distance_matrix[unvisited_nodes, destination_node]
    sub_dist[n_unvisited, :n_unvisited] = distance_matrix[destination_node, unvisited_nodes]
    sub_dist[n_unvisited, n_unvisited] = 0.0
    
    # Prim's algorithm to build MST and track degrees
    in_mst = np.zeros(n_total, dtype=bool)
    in_mst[0] = True
    min_dist = sub_dist[0].copy()
    parent = np.full(n_total, -1, dtype=int)
    
    degree = np.zeros(n_total, dtype=int)
    
    for _ in range(n_total - 1):
        # Find node with minimum distance to MST
        # Mask out already-in-MST nodes
        mask = ~in_mst
        min_dist[mask] = np.where(mask, min_dist, np.inf)
        next_node = np.argmin(min_dist)
        
        if parent[next_node] != -1:
            degree[next_node] += 1
            degree[parent[next_node]] += 1
        
        in_mst[next_node] = True
        parent[next_node] = parent[next_node]  # keep as is
        
        # Update min_dist
        for i in range(n_total):
            if not in_mst[i] and sub_dist[next_node, i] < min_dist[i]:
                min_dist[i] = sub_dist[next_node, i]
                parent[i] = next_node
    
    # Leaves are nodes with degree 1 (excluding destination which is index n_unvisited)
    # We only care about unvisited nodes (indices 0..n_unvisited-1)
    leaf_mask = (degree[:n_unvisited] <= 1)
    
    # If no leaves (shouldn't happen), fall back to all
    if not np.any(leaf_mask):
        leaf_mask = np.ones(n_unvisited, dtype=bool)
    
    return leaf_mask


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using an MST-leaf-pruning heuristic combined with
    immediate and lookahead distance costs.
    
    The heuristic:
    1. Compute MST over (unvisited + destination) to identify leaves
    2. Score candidates as: w1 * dist(current, cand) + w2 * dist(cand, dest)
    3. Apply a bonus to MST leaves to prefer visiting them early
       (they must be connected in the tour, and visiting them now
       reduces future connection costs)
    """
    # Safety check: if no unvisited nodes, return -1
    if len(unvisited_nodes) == 0:
        return -1
    
    # Check deadline - if approaching, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    
    # Compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from each unvisited to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute MST leaves
    leaf_mask = _compute_mst_leaves(unvisited_nodes, destination_node, distance_matrix)
    
    # Base score: weighted sum of immediate and lookahead
    alpha = 1.0
    beta = 0.3
    scores = alpha * dist_from_current + beta * dist_to_dest
    
    # Apply leaf bonus: reduce score for MST leaves
    # This encourages visiting leaves early, following the global skeleton
    # The bonus is proportional to the distance from current to the leaf,
    # scaled by a factor that makes it meaningful but not dominant
    leaf_bonus = 0.15 * dist_from_current
    scores[leaf_mask] -= leaf_bonus[leaf_mask]
    
    # Select the candidate with minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
