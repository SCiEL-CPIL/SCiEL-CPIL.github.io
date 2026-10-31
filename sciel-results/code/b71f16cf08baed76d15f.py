"""Model-and-optimize TSP constructive heuristic with MST-based residual cost.

Strategy:
  1. For each new instance, fit parameters (alpha, beta, gamma) via a small
     grid search using a fast approximate tour builder.
  2. Use a vectorized scoring function that combines:
     - direct distance from current node
     - distance to destination (to avoid long final edge)
     - MST-based residual cost reduction (global structural awareness)
  3. Add a smooth connectivity penalty.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest)
             + beta * (MST_weight_before - MST_weight_after_removing_c)
             + gamma * (d(c,dest) - min_u d(u,dest))  [isolation penalty]

The MST term captures how much the global connectivity cost decreases
when we commit to visiting c next. This is a principled surrogate for
the remaining tour cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 145.0  # Safety margin before 150s limit

# Per-instance state
_FIT_PARAMS = None  # (alpha, beta, gamma) best found by fitting
_instance_id = -1
_last_unvisited_len = -1


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _mst_weight(dist_matrix_subset):
    """Compute MST weight of a complete graph given by distance matrix.
    Uses Prim's algorithm in O(n^2) for dense graphs."""
    n = dist_matrix_subset.shape[0]
    if n <= 1:
        return 0.0
    if n == 2:
        return float(dist_matrix_subset[0, 1])
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total = 0.0
    
    for _ in range(n):
        # Find node with minimum distance not yet in MST
        mask = ~in_mst
        min_dist[mask] = np.where(mask, min_dist, np.inf)
        u = int(np.argmin(min_dist))
        in_mst[u] = True
        total += min_dist[u]
        
        # Update distances
        if u < n - 1:
            new_dists = dist_matrix_subset[u, in_mst == False]
            idxs = np.where(in_mst == False)[0]
            min_dist[idxs] = np.minimum(min_dist[idxs], new_dists)
    
    return total


def _mst_weight_fast(dist_matrix, node_indices):
    """Compute MST weight for a subset of nodes using the full distance matrix."""
    n = len(node_indices)
    if n <= 1:
        return 0.0
    if n == 2:
        return float(dist_matrix[node_indices[0], node_indices[1]])
    
    # Build submatrix
    sub = dist_matrix[np.ix_(node_indices, node_indices)]
    return _mst_weight(sub)


def _mst_reduction_score(dist_matrix, current_node, unvisited_nodes, destination_node):
    """For each candidate c in unvisited_nodes, compute the MST weight reduction
    when c is removed from the remaining set.
    
    MST_weight_before = MST(all remaining nodes)
    MST_weight_after_c = MST(remaining \\ {c})
    reduction(c) = MST_weight_before - MST_weight_after_c
    
    Higher reduction means removing c (visiting it) reduces the residual
    connectivity cost more, so it's a better choice.
    """
    n_rem = len(unvisited_nodes)
    if n_rem <= 2:
        return np.zeros(n_rem)
    
    # Compute MST weight of all remaining nodes
    mst_before = _mst_weight_fast(dist_matrix, unvisited_nodes)
    
    # For each candidate, compute MST weight of remaining without it
    reductions = np.zeros(n_rem)
    for i in range(n_rem):
        # Remove node i from unvisited
        if n_rem == 3:
            # Only 2 nodes left, MST is just the edge between them
            remaining_idx = [j for j in range(n_rem) if j != i]
            mst_after = float(dist_matrix[unvisited_nodes[remaining_idx[0]], unvisited_nodes[remaining_idx[1]]])
        else:
            remaining_nodes = np.delete(unvisited_nodes, i)
            mst_after = _mst_weight_fast(dist_matrix, remaining_nodes)
        reductions[i] = mst_before - mst_after
    
    return reductions


def _build_tour(distance_matrix, start_node, alpha=0.0, beta=0.0, gamma=0.0, use_mst=True):
    """Build a full TSP tour using the given parameters. Returns tour length."""
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    visited_mask = np.zeros(n, dtype=bool)
    visited_mask[0] = True
    
    while len(remaining) > 0:
        n_rem = len(remaining)
        if n_rem == 1:
            next_node = remaining[0]
        else:
            # Compute scores
            d_cur = distance_matrix[current, remaining]
            scores = d_cur.copy()
            
            if alpha > 0:
                d_dest = distance_matrix[remaining, start_node]
                scores += alpha * d_dest
            
            if beta > 0 and n_rem > 2 and use_mst:
                # MST-based reduction score
                reductions = _mst_reduction_score(distance_matrix, current, np.array(remaining), start_node)
                # Higher reduction is better, so subtract
                scores -= beta * reductions
            
            if gamma > 0 and n_rem > 2:
                # Isolation penalty
                d_dest_all = distance_matrix[remaining, start_node]
                min_d_dest = np.min(d_dest_all)
                scores += gamma * (d_dest_all - min_d_dest)
            
            best_idx = np.argmin(scores)
            next_node = remaining[best_idx]
        
        total_dist += distance_matrix[current, next_node]
        visited_mask[next_node] = True
        remaining.remove(next_node)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=2.0):
    """Fit alpha, beta, gamma by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Grid of parameters
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.25, 0.4]
    beta_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.2]
    gamma_values = [0.0, 0.05, 0.1, 0.2]
    
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    deadline = elapsed_start + min(time_budget, max(0.5, _LIMIT - (time.time() - _start_time) - 5.0))
    
    # Try a subset of combinations for speed
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for beta in beta_values:
            if time.time() > deadline:
                break
            for gamma in gamma_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour(distance_matrix, start_node, alpha, beta, gamma, use_mst=True)
                if tour_len < best_len:
                    best_len = tour_len
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
    
    _FIT_PARAMS = (best_alpha, best_beta, best_gamma)
    return (best_alpha, best_beta, best_gamma)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    """
    global _FIT_PARAMS, _instance_id, _last_unvisited_len
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)
    
    if is_new_instance:
        _instance_id += 1
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=2.5)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, beta, gamma = 0.1, 0.5, 0.1
    else:
        alpha, beta, gamma = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized where possible)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest
    
    if beta > 0 and n_unvisited > 2:
        # MST-based reduction score
        reductions = _mst_reduction_score(distance_matrix, current_node, unvisited_nodes, destination_node)
        scores -= beta * reductions
    
    if gamma > 0 and n_unvisited > 2:
        # Isolation penalty
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += gamma * (d_dest_all - min_d_dest)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
