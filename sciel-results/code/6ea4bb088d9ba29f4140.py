"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta, temp) via a
     small grid search using a fast approximate tour builder.
  2. Use a vectorized 2-step lookahead with soft-min (log-sum-exp) for
     smoothness.
  3. Add a smooth connectivity/isolation penalty term.
  4. Critically: reset per-instance state when a new instance is detected,
     so parameters are not carried over between different problem instances.
  5. Optimize the fitting loop to be faster by reducing the grid and
     short-circuiting when time is tight.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*softmin_{u in rem\\{c}} d(c,u)
             + delta * (d(c,dest) - softmin_u d(u,dest))  [isolation penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state - MUST be reset between instances
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, gamma, delta, temp) best found by fitting
_instance_counter = 0
_last_dm_shape = None  # Track distance matrix shape to detect new instances
_last_dm_checksum = None  # Quick checksum to detect if same matrix


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _softmin_vectorized(arr_2d, temperature=1.0):
    """Vectorized soft-min along axis=1 for a 2D array.
    
    Returns an array where each element is the soft-min of the corresponding row.
    softmin(x) = -T * log(sum(exp(-x_i / T)))
    """
    if arr_2d.ndim == 1:
        arr_2d = arr_2d.reshape(1, -1)
    if arr_2d.size == 0:
        return np.zeros(arr_2d.shape[0])
    if temperature <= 0:
        return np.min(arr_2d, axis=1)
    # Numerically stable: subtract max before exp
    scaled = -arr_2d / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    # Handle cases where max is -inf (all -inf) or +inf
    max_scaled_safe = np.where(np.isfinite(max_scaled), max_scaled, 0)
    exp_vals = np.exp(scaled - max_scaled_safe)
    # Where max was not finite, the row was all -inf or had issues; fall back to min
    log_sum = np.log(np.sum(exp_vals, axis=1, keepdims=True))
    result = -temperature * (max_scaled_safe.squeeze() + log_sum.squeeze())
    # For rows where max_scaled was not finite, use hard min
    bad_mask = ~np.isfinite(max_scaled.squeeze())
    if np.any(bad_mask):
        result[bad_mask] = np.min(arr_2d[bad_mask], axis=1)
    return result


def _softmin_1d(arr, temperature=1.0):
    """Soft-min for a 1D array."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    if temperature <= 0:
        return float(np.min(arr))
    scaled = -arr / temperature
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return float(np.min(arr))
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return float(-temperature * log_sum)


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0, 
                     soft_temp=0.5, use_soft=False, epsilon=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length.
    
    Optimized for speed: uses vectorized operations and avoids unnecessary work.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        # Vectorized scoring
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest
        
        if gamma > 0 and len(remaining) > 2:
            # For each candidate c, compute min distance from c to other remaining nodes
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)].copy()
            np.fill_diagonal(sub_dist, np.inf)
            if use_soft:
                min_next = _softmin_vectorized(sub_dist, temperature=soft_temp)
            else:
                min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 2:
            # Isolation penalty: penalize candidates whose distance to dest
            # is much larger than the minimum among remaining
            d_dest_all = distance_matrix[rem_arr, start_node]
            if use_soft:
                min_d_dest = _softmin_1d(d_dest_all, temperature=soft_temp)
            else:
                min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        if epsilon > 0 and len(remaining) > 3:
            # 3-step lookahead: for each candidate c, estimate cost of the next 2 edges
            # via soft-min of (d(c,u) + min_{v != c,u} d(u,v))
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)].copy()
            np.fill_diagonal(sub_dist, np.inf)
            if use_soft:
                min_next = _softmin_vectorized(sub_dist, temperature=soft_temp)
            else:
                min_next = np.min(sub_dist, axis=1)
            # For 3-step: d(c,u) + d(u,v) approximated by d(c,u) + min_v d(u,v)
            # We already have min_next[c] = min_u d(c,u), but we want sum over u of d(c,u)+min_v d(u,v)
            # Approximate: for each c, soft-min over u of (d(c,u) + min_v d(u,v))
            # min_v d(u,v) for each u is min_next[u]
            # So we need: for each c, soft-min over u of (d(c,u) + min_next[u])
            # This is a weighted combination
            combined = sub_dist + min_next[np.newaxis, :]  # shape (n_rem, n_rem), row c, col u
            if use_soft:
                three_step = _softmin_vectorized(combined, temperature=soft_temp)
            else:
                three_step = np.min(combined, axis=1)
            scores += epsilon * three_step
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta, temp, epsilon by grid search on the full instance.
    
    Uses fast tour builder for speed. Tries hard-min first, then refines
    with soft-min if time permits.
    """
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.5, 0.0)
    
    # Focused grid for speed
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    delta_values = [0.0, 0.05, 0.1]
    epsilon_values = [0.0, 0.05, 0.1]
    temp_values = [0.5, 1.0]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_temp = 0.5
    best_epsilon = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    # Phase 1: Coarse grid with hard min (fast) - skip temp variation
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                for epsilon in epsilon_values:
                    if time.time() > deadline:
                        break
                    tour_len = _build_tour_fast(distance_matrix, start_node, alpha, gamma, delta, 
                                                use_soft=False, epsilon=epsilon)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_alpha = alpha
                        best_gamma = gamma
                        best_delta = delta
                        best_epsilon = epsilon
    
    # Phase 2: If time remains, try soft-min around best point with varying temperature
    if time.time() < deadline - 0.5:
        for temp in temp_values:
            for sa in [max(0, best_alpha - 0.05), best_alpha, min(0.3, best_alpha + 0.05)]:
                if time.time() > deadline:
                    break
                for sg in [max(0, best_gamma - 0.05), best_gamma, min(0.3, best_gamma + 0.05)]:
                    if time.time() > deadline:
                        break
                    for sd in [max(0, best_delta - 0.05), best_delta, min(0.15, best_delta + 0.05)]:
                        if time.time() > deadline:
                            break
                        for se in [max(0, best_epsilon - 0.05), best_epsilon, min(0.2, best_epsilon + 0.05)]:
                            if time.time() > deadline:
                                break
                            tour_len = _build_tour_fast(distance_matrix, start_node, sa, sg, sd,
                                                        soft_temp=temp, use_soft=True, epsilon=se)
                            if tour_len < best_len:
                                best_len = tour_len
                                best_alpha = sa
                                best_gamma = sg
                                best_delta = sd
                                best_epsilon = se
                                best_temp = temp
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_delta, best_temp, best_epsilon)
    return (best_alpha, best_gamma, best_delta, best_temp, best_epsilon)


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance by checking unvisited count 
    against matrix size and using a checksum."""
    global _last_dm_shape, _last_dm_checksum
    
    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)
    
    # A new instance starts when all (or nearly all) nodes are unvisited
    # and the distance matrix is different from the previous one
    if n_unvisited >= n_total - 1:
        # Quick check: is this the same matrix as before?
        # Use a small checksum for speed
        checksum = np.sum(distance_matrix[::max(1, n_total//10), ::max(1, n_total//10)])
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_shape = distance_matrix.shape
            _last_dm_checksum = checksum
            return True
        elif _last_dm_shape != distance_matrix.shape:
            _last_dm_shape = distance_matrix.shape
            _last_dm_checksum = checksum
            return True
    
    return False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    Critically resets parameters when a new instance is detected.
    """
    global _current_instance_id, _FIT_PARAMS, _instance_counter, _last_dm_shape, _last_dm_checksum
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)
    
    if is_new_instance:
        _instance_counter += 1
        # Reset parameters for the new instance
        _FIT_PARAMS = None
        # Use more time for first instance, less for subsequent
        tb = 3.0 if _instance_counter == 1 else 1.5
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=tb)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, gamma, delta, temp, epsilon = 0.1, 0.1, 0.05, 0.5, 0.0
    else:
        alpha, gamma, delta, temp, epsilon = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        # Vectorized 2-step lookahead with soft-min for smoothness
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        min_next = _softmin_vectorized(sub_dist, temperature=temp)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        # Isolation penalty with soft-min
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = _softmin_1d(d_dest_all, temperature=temp)
        scores += delta * (d_dest_all - min_d_dest)
    
    if epsilon > 0 and n_unvisited > 3:
        # 3-step lookahead: for each candidate c, estimate cost of next 2 edges
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        min_next = _softmin_vectorized(sub_dist, temperature=temp)
        combined = sub_dist + min_next[np.newaxis, :]
        three_step = _softmin_vectorized(combined, temperature=temp)
        scores += epsilon * three_step
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
