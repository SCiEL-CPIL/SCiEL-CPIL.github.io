"""Model-and-optimize TSP constructive heuristic with minimax bottleneck focus.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta, epsilon) via
     grid search, with emphasis on the minimax (bottleneck) term epsilon.
  2. Use soft-min/soft-max (log-sum-exp) for smoothness in scoring.
  3. The minimax penalty discourages selecting nodes far from the remaining
     cluster, preventing any single long edge in the final tour.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*softmin_{u in rem\{c}} d(c,u)
             + delta * (d(c,dest) - softmin_u d(u,dest))
             + epsilon * softmax_{u in rem\{c}} d(c,u)   [bottleneck penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_FIT_PARAMS = None  # (alpha, gamma, delta, epsilon) best found by fitting
_instance_counter = 0
_soft_temp = 0.4  # Temperature for soft-min/soft-max operations


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _softmin(arr, temperature=0.4):
    """Approximate min via log-sum-exp for smoothness."""
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


def _softmin_vectorized(matrix, temperature=0.4):
    """Vectorized soft-min along axis=1 for a 2D matrix."""
    if matrix.size == 0:
        return np.zeros(matrix.shape[0])
    if matrix.shape[1] == 1:
        return matrix[:, 0].copy()
    if temperature <= 0:
        return np.min(matrix, axis=1)
    scaled = -matrix / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    if not np.all(np.isfinite(max_scaled)):
        return np.min(matrix, axis=1)
    exp_scaled = np.exp(scaled - max_scaled)
    log_sum = max_scaled.squeeze() + np.log(np.sum(exp_scaled, axis=1))
    return -temperature * log_sum


def _softmax_vectorized(matrix, temperature=0.4):
    """Vectorized soft-max along axis=1 for a 2D matrix.
    
    softmax(x) = T * log(sum(exp(x_i / T)))
    As T -> 0, approaches max(x). For larger T, smoother.
    """
    if matrix.size == 0:
        return np.zeros(matrix.shape[0])
    if matrix.shape[1] == 1:
        return matrix[:, 0].copy()
    if temperature <= 0:
        return np.max(matrix, axis=1)
    scaled = matrix / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    if not np.all(np.isfinite(max_scaled)):
        return np.max(matrix, axis=1)
    exp_scaled = np.exp(scaled - max_scaled)
    log_sum = max_scaled.squeeze() + np.log(np.sum(exp_scaled, axis=1))
    return temperature * log_sum


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0,
                     epsilon=0.0, soft_temp=0.4):
    """Build a full TSP tour using the given parameters. Returns tour length.
    
    Uses hard min/max for speed during fitting.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest
        
        if gamma > 0 and len(remaining) > 2:
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 2:
            d_dest_all = distance_matrix[rem_arr, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        if epsilon > 0 and len(remaining) > 2:
            # Minimax: penalize candidates whose max distance to other remaining is large
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, 0.0)
            max_next = np.max(sub_dist, axis=1)
            scores += epsilon * max_next
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta, epsilon by grid search on the full instance.
    
    Emphasizes the minimax (epsilon) dimension since bottleneck cost is critical.
    """
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.0)
    
    # Grid with more emphasis on epsilon (minimax) values
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.25]
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.25]
    delta_values = [0.0, 0.03, 0.07, 0.12]
    epsilon_values = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_epsilon = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
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
                    tour_len = _build_tour_fast(distance_matrix, start_node, 
                                                alpha, gamma, delta, epsilon)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_alpha = alpha
                        best_gamma = gamma
                        best_delta = delta
                        best_epsilon = epsilon
    
    # Fine-tune around best found
    if time.time() < deadline - 1.0:
        sa_vals = [max(0, best_alpha - 0.03), best_alpha, min(0.4, best_alpha + 0.03)]
        sg_vals = [max(0, best_gamma - 0.03), best_gamma, min(0.35, best_gamma + 0.03)]
        sd_vals = [max(0, best_delta - 0.02), best_delta, min(0.2, best_delta + 0.02)]
        se_vals = [max(0, best_epsilon - 0.03), best_epsilon, min(0.4, best_epsilon + 0.03)]
        
        for sa in sa_vals:
            if time.time() > deadline:
                break
            for sg in sg_vals:
                if time.time() > deadline:
                    break
                for sd in sd_vals:
                    if time.time() > deadline:
                        break
                    for se in se_vals:
                        if time.time() > deadline:
                            break
                        tour_len = _build_tour_fast(distance_matrix, start_node,
                                                    sa, sg, sd, se)
                        if tour_len < best_len:
                            best_len = tour_len
                            best_alpha = sa
                            best_gamma = sg
                            best_delta = sd
                            best_epsilon = se
    
    return (best_alpha, best_gamma, best_delta, best_epsilon)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize with minimax bottleneck focus:
    - Fits parameters per instance via grid search
    - Uses soft-min/soft-max for smooth scoring
    - Minimax penalty prevents long edges in the final tour
    """
    global _FIT_PARAMS, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)
    
    if is_new_instance:
        _instance_counter += 1
        tb = 3.5 if _instance_counter == 1 else 2.5
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=tb)
    
    if _FIT_PARAMS is None:
        alpha, gamma, delta, epsilon = 0.08, 0.08, 0.05, 0.10
    else:
        alpha, gamma, delta, epsilon = _FIT_PARAMS
    
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    unvisited_arr = np.asarray(unvisited_nodes)
    dists_from_current = distance_matrix[current_node, unvisited_arr]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_arr, destination_node]
        scores += alpha * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = _softmin_vectorized(sub_dist, temperature=_soft_temp)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        d_dest_all = distance_matrix[unvisited_arr, destination_node]
        min_d_dest = _softmin(d_dest_all, temperature=_soft_temp)
        scores += delta * (d_dest_all - min_d_dest)
    
    if epsilon > 0 and n_unvisited > 2:
        # Minimax bottleneck penalty using soft-max for smoothness
        sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        np.fill_diagonal(sub_dist, 0.0)
        max_next = _softmax_vectorized(sub_dist, temperature=_soft_temp)
        scores += epsilon * max_next
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_arr[best_idx])

# EVOLVE-BLOCK-END
