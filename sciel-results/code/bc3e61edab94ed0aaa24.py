"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, eta, delta) via a
     small grid search using a fast approximate tour builder.
  2. Use a vectorized multi-step lookahead (2-step + 3-step) for efficiency.
  3. Add a smooth connectivity/isolation penalty term.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*min_{u in rem\\{c}} d(c,u)
             + eta*min_{v in rem\\{c}} [d(c,v) + min_{w in rem\\{c,v}} d(v,w)]
             + delta * (d(c,dest) - min_u d(u,dest))  [isolation penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_FIT_PARAMS = None  # (alpha, gamma, eta, delta) best found by fitting
_instance_counter = 0
_last_instance_dm_id = -1  # track which distance matrix we last fitted


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0, eta=0.0, delta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length."""
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))
    
    while remaining:
        rem_arr = np.array(remaining, dtype=np.intp)
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest
        
        if (gamma > 0 or eta > 0) and len(remaining) > 2:
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, np.inf)
            
            if gamma > 0:
                min_next = np.min(sub_dist, axis=1)
                scores += gamma * min_next
            
            if eta > 0:
                # 3-step lookahead: for each candidate c, find cheapest v != c
                # such that d(c,v) + min_{w != c,v} d(v,w) is small
                # Approximate: for each c, take min over v of (d(c,v) + min_next_v)
                # where min_next_v = min_{w != c,v} d(v,w)
                # We already have sub_dist (remaining x remaining with diag=inf)
                # For row c, min_next is min over all other nodes
                # For 3-step: for each c, compute min_v [d(c,v) + min_{w != c,v} d(v,w)]
                # min_{w != c,v} d(v,w) = second smallest in row v excluding c
                # Approximation: use min_next[v] as proxy (it excludes v itself, 
                # which is correct; it may include c but that's a minor approximation)
                min_next_all = np.min(sub_dist, axis=1)  # min over all other nodes for each row
                # For each c: min over v of (sub_dist[c,v] + min_next_all[v])
                # This is a matrix product-like operation
                # scores_3step[c] = min_v (sub_dist[c,v] + min_next_all[v])
                # = min over columns of (sub_dist + min_next_all[None, :])
                adjusted = sub_dist + min_next_all[np.newaxis, :]
                scores_3step = np.min(adjusted, axis=1)
                scores += eta * scores_3step
        
        if delta > 0 and len(remaining) > 2:
            d_dest_all = distance_matrix[rem_arr, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, eta, delta by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.0)
    
    # Grid of parameters — expanded to include eta (3-step lookahead)
    alpha_values = [0.0, 0.03, 0.07, 0.12, 0.2, 0.35]
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.25]
    eta_values = [0.0, 0.03, 0.07, 0.12]
    delta_values = [0.0, 0.05, 0.15]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_eta = 0.0
    best_delta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_budget = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.3, remaining_budget))
    
    # To keep the grid manageable, we do a two-phase search:
    # Phase 1: coarse grid (subset)
    # Phase 2: fine grid around best
    
    # Phase 1: coarse
    alpha_coarse = [0.0, 0.07, 0.2, 0.35]
    gamma_coarse = [0.0, 0.1, 0.25]
    eta_coarse = [0.0, 0.07]
    delta_coarse = [0.0, 0.05]
    
    for alpha in alpha_coarse:
        if time.time() > deadline:
            break
        for gamma in gamma_coarse:
            if time.time() > deadline:
                break
            for eta in eta_coarse:
                if time.time() > deadline:
                    break
                for delta in delta_coarse:
                    if time.time() > deadline:
                        break
                    tour_len = _build_tour_fast(distance_matrix, start_node, alpha, gamma, eta, delta)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_alpha = alpha
                        best_gamma = gamma
                        best_eta = eta
                        best_delta = delta
    
    # Phase 2: fine grid around best (if time remains)
    if time.time() < deadline - 0.5:
        alpha_fine = [max(0, best_alpha - 0.05), best_alpha, min(0.5, best_alpha + 0.05)]
        gamma_fine = [max(0, best_gamma - 0.05), best_gamma, min(0.4, best_gamma + 0.05)]
        eta_fine = [max(0, best_eta - 0.04), best_eta, min(0.2, best_eta + 0.04)]
        delta_fine = [max(0, best_delta - 0.03), best_delta, min(0.2, best_delta + 0.03)]
        
        for alpha in alpha_fine:
            if time.time() > deadline:
                break
            for gamma in gamma_fine:
                if time.time() > deadline:
                    break
                for eta in eta_fine:
                    if time.time() > deadline:
                        break
                    for delta in delta_fine:
                        if time.time() > deadline:
                            break
                        tour_len = _build_tour_fast(distance_matrix, start_node, alpha, gamma, eta, delta)
                        if tour_len < best_len:
                            best_len = tour_len
                            best_alpha = alpha
                            best_gamma = gamma
                            best_eta = eta
                            best_delta = delta
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_eta, best_delta)
    return (best_alpha, best_gamma, best_eta, best_delta)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    """
    global _FIT_PARAMS, _instance_counter, _last_instance_dm_id
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance: check if this is the first call for a new distance matrix
    # We use the id of the distance_matrix array as a proxy for instance identity.
    # Also detect by unvisited count being close to total (first/second step).
    n_total = distance_matrix.shape[0]
    dm_id = id(distance_matrix)
    
    is_new_instance = (n_unvisited >= n_total - 1) and (dm_id != _last_instance_dm_id)
    
    if is_new_instance:
        _last_instance_dm_id = dm_id
        _instance_counter += 1
        # Scale time budget: give more time to earlier instances
        time_budget = 3.0 if _instance_counter <= 2 else 2.0
        if _instance_counter >= 6:
            time_budget = 1.5
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=time_budget)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, gamma, eta, delta = 0.05, 0.0, 0.0, 0.0
    else:
        alpha, gamma, eta, delta = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    rem_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    dists_from_current = distance_matrix[current_node, rem_arr]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[rem_arr, destination_node]
        scores += alpha * dists_to_dest
    
    if (gamma > 0 or eta > 0) and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
        np.fill_diagonal(sub_dist, np.inf)
        
        if gamma > 0:
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if eta > 0:
            min_next_all = np.min(sub_dist, axis=1)
            adjusted = sub_dist + min_next_all[np.newaxis, :]
            scores_3step = np.min(adjusted, axis=1)
            scores += eta * scores_3step
    
    if delta > 0 and n_unvisited > 2:
        d_dest_all = distance_matrix[rem_arr, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
