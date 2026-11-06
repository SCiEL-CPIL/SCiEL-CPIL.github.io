"""TSP constructive heuristic with global structural awareness."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = None
_DEADLINE = 145.0


def _remaining_time():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return _DEADLINE - (time.time() - _T0)


def _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix):
    if unvisited_nodes.size == 0:
        return int(current_node)
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])


def _mst_cost_fast(nodes, distance_matrix):
    """Fast Prim's MST cost on the given node set."""
    n = nodes.size
    if n <= 1:
        return 0.0
    visited = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total = 0.0
    for _ in range(n):
        # Find unvisited node with smallest min_dist
        unvis_mask = ~visited
        if not np.any(unvis_mask):
            break
        idx = int(np.argmin(np.where(unvis_mask, min_dist, np.inf)))
        if min_dist[idx] == np.inf:
            break
        visited[idx] = True
        total += min_dist[idx]
        # Update distances
        d_from_idx = distance_matrix[nodes[idx], nodes]
        update_mask = unvis_mask & ~visited
        min_dist[update_mask] = np.minimum(min_dist[update_mask], d_from_idx[update_mask])
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = unvisited_nodes.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # If time is running out, fall back to cheap nearest-neighbour.
    if _remaining_time() < 0.05:
        return _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix)

    # For small remaining sets, use more careful evaluation
    if n <= 8:
        # Compute insertion cost for each candidate
        best_score = np.inf
        best_idx = 0
        d_cur = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # For small n, do a simple 2-step lookahead
        for i in range(n):
            # Cost if we pick node i next: d_cur[i] + best next from remaining
            remaining = np.delete(unvisited_nodes, i)
            if remaining.size == 0:
                est_rest = 0.0
            else:
                # Approximate remaining tour cost via MST + endpoint correction
                d_rem = distance_matrix[remaining, remaining]
                mst = 0.0
                if remaining.size > 1:
                    mst = _mst_cost_fast(remaining, distance_matrix)
                # Add connections from current->remaining->destination
                # This is a rough proxy
                d_to_rem = distance_matrix[current_node, remaining]
                d_rem_to_dest = distance_matrix[remaining, destination_node]
                # Minimum spanning tree already covers internal edges
                # We need to connect current and destination
                min_from_cur = np.min(d_to_rem) if remaining.size > 0 else 0.0
                min_to_dest = np.min(d_rem_to_dest) if remaining.size > 0 else 0.0
                est_rest = mst + min_from_cur + min_to_dest
            
            score = d_cur[i] + est_rest
            if score < best_score:
                best_score = score
                best_idx = i
        return int(unvisited_nodes[best_idx])

    # For larger n, use a hybrid scoring: immediate + destination + structural
    d_cur = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute a global structural score: for each candidate, estimate the
    # MST cost of the remaining unvisited nodes after removing it
    # This is expensive, so we use a cheaper proxy: 
    # the average distance from each candidate to other unvisited nodes
    # captures "centrality" - central nodes are better to visit early
    
    # Compute pairwise distances among unvisited
    d_unvis = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Average distance from each node to all others (lower = more central)
    avg_dist = np.mean(d_unvis, axis=1)
    
    # Soft-min of d_cur (smooth nearest)
    temp = 0.3
    neg = -d_cur / temp
    m = np.max(neg)
    soft_min_cur = -(m + np.log(np.sum(np.exp(neg - m)))) * temp
    
    # Combine scores with weights:
    # - Immediate edge (dominant)
    # - Closure to destination (important for final leg)
    # - Centrality (prefer visiting central nodes early to avoid stranding)
    # - Soft-min as tie-breaker
    w_immediate = 1.0
    w_closure = 0.4
    w_central = 0.15
    w_softmin = 0.1
    
    score = (w_immediate * d_cur + 
             w_closure * d_dest + 
             w_central * avg_dist + 
             w_softmin * soft_min_cur)
    
    idx = int(np.argmin(score))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
