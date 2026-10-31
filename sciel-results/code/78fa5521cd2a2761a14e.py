"""Model-and-optimize TSP constructive heuristic combining potential field repulsion
with smooth 2-step lookahead and isolation penalty.

Strategy:
  1. For each new instance, fit parameters (lambda, mu, gamma, delta) via grid search.
  2. Use a combined scoring function:
     score(c) = d(current, c) 
                + lambda * sum(1 / (d(c, u) + eps) for u in unvisited, u != c)
                + mu * d(c, destination)
                + gamma * min_{u in rem\{c}} d(c, u)   [2-step lookahead / connectivity]
                + delta * (d(c, dest) - min_u d(u, dest)) [isolation penalty]
  3. The repulsive potential encourages spatial coverage by penalizing nodes that are
     too close to many other unvisited nodes. The min-distance term encourages 2-step
     lookahead. The isolation penalty prevents long edges near the end of the tour.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_FIT_PARAMS = None  # (lambda, mu, gamma, delta) best found by fitting
_instance_counter = 0
_dist_epsilon = 1e-6  # Small epsilon for numerical stability in 1/d


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _compute_potential(unvisited_arr, distance_matrix, epsilon=1e-6):
    """Compute the repulsive potential for each unvisited node.
    P(c) = sum_{u in U, u != c} 1 / (d(c, u) + epsilon)
    """
    n = len(unvisited_arr)
    if n <= 1:
        return np.zeros(n)
    
    # Get submatrix of distances between unvisited nodes
    D_sub = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
    
    # Compute inverse distances
    # Add epsilon to avoid division by zero
    inv_D = 1.0 / (D_sub + epsilon)
    
    # Set diagonal to 0 (self-distance)
    np.fill_diagonal(inv_D, 0.0)
    
    # Sum along axis 1 to get potential for each node
    potentials = np.sum(inv_D, axis=1)
    
    return potentials


def _build_tour_fast(distance_matrix, start_node, lam=0.0, mu=0.0, gamma=0.0, delta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length.
    
    Uses hard min for speed during fitting.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        d_cur = distance_matrix[current, rem_arr]
        
        scores = d_cur.copy()
        
        if lam > 0 and len(remaining) > 1:
            potentials = _compute_potential(rem_arr, distance_matrix)
            scores += lam * potentials
        
        if mu > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += mu * d_dest
        
        if gamma > 0 and len(remaining) > 2:
            # 2-step lookahead: min distance from candidate to other remaining nodes
            sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)]
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and len(remaining) > 2:
            # Isolation penalty: penalize candidates whose distance to dest
            # is much larger than the minimum among remaining
            d_dest_all = distance_matrix[rem_arr, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        best_idx = int(np.argmin(scores))
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit lambda, mu, gamma, delta by grid search on the full instance."""
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.0)
    
    # Grid search over lambda, mu, gamma, delta
    # Start with coarser grid for initial exploration
    lambda_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    mu_values = [0.0, 0.05, 0.1, 0.2, 0.5]
    gamma_values = [0.0, 0.05, 0.1, 0.2]
    delta_values = [0.0, 0.05, 0.15]
    
    best_lambda = 0.0
    best_mu = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))
    
    # First pass: explore the full grid
    for lam in lambda_values:
        if time.time() > deadline:
            break
        for mu in mu_values:
            if time.time() > deadline:
                break
            for gamma in gamma_values:
                if time.time() > deadline:
                    break
                for delta in delta_values:
                    if time.time() > deadline:
                        break
                    tour_len = _build_tour_fast(distance_matrix, start_node, lam, mu, gamma, delta)
                    if tour_len < best_len:
                        best_len = tour_len
                        best_lambda = lam
                        best_mu = mu
                        best_gamma = gamma
                        best_delta = delta
    
    # Fine-tune around best found if time permits
    if time.time() < deadline - 1.0:
        # Small steps around the best found values
        step_l = max(0.01, best_lambda * 0.1) if best_lambda > 0 else 0.01
        step_m = max(0.01, best_mu * 0.1) if best_mu > 0 else 0.01
        step_g = max(0.01, best_gamma * 0.1) if best_gamma > 0 else 0.01
        step_d = max(0.01, best_delta * 0.1) if best_delta > 0 else 0.01
        
        lam_vals = [max(0, best_lambda - step_l), best_lambda, best_lambda + step_l]
        mu_vals = [max(0, best_mu - step_m), best_mu, best_mu + step_m]
        gamma_vals = [max(0, best_gamma - step_g), best_gamma, best_gamma + step_g]
        delta_vals = [max(0, best_delta - step_d), best_delta, best_delta + step_d]
        
        for lam in lam_vals:
            if time.time() > deadline:
                break
            for mu in mu_vals:
                if time.time() > deadline:
                    break
                for gamma in gamma_vals:
                    if time.time() > deadline:
                        break
                    for delta in delta_vals:
                        if time.time() > deadline:
                            break
                        tour_len = _build_tour_fast(distance_matrix, start_node, lam, mu, gamma, delta)
                        if tour_len < best_len:
                            best_len = tour_len
                            best_lambda = lam
                            best_mu = mu
                            best_gamma = gamma
                            best_delta = delta
    
    return (best_lambda, best_mu, best_gamma, best_delta)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses model-and-optimize with combined potential field repulsion,
    2-step lookahead, and isolation penalty:
    - Fits parameters (lambda, mu, gamma, delta) per instance via grid search
    - Uses sum of inverse distances as repulsive potential
    - Uses min distance to remaining nodes for 2-step lookahead
    - Uses isolation penalty to prevent long edges near tour end
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
        lam, mu, gamma, delta = 0.1, 0.1, 0.0, 0.0
    else:
        lam, mu, gamma, delta = _FIT_PARAMS
    
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    unvisited_arr = np.asarray(unvisited_nodes)
    dists_from_current = distance_matrix[current_node, unvisited_arr]
    scores = dists_from_current.copy()
    
    if lam > 0 and n_unvisited > 1:
        potentials = _compute_potential(unvisited_arr, distance_matrix)
        scores += lam * potentials
    
    if mu > 0:
        dists_to_dest = distance_matrix[unvisited_arr, destination_node]
        scores += mu * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        # 2-step lookahead: min distance from candidate to other remaining nodes
        sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        # Isolation penalty: penalize candidates whose distance to dest
        # is much larger than the minimum among remaining
        d_dest_all = distance_matrix[unvisited_arr, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_arr[best_idx])

# EVOLVE-BLOCK-END
