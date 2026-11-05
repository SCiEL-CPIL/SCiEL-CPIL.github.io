"""CVRP heuristic: savings + regret + smooth continuation cost + capacity fill."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Calculate Savings
    # Savings = d[current, depot] + d[depot, c] - d[current, c]
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Smooth continuation cost using soft-min (log-sum-exp trick)
    # For each candidate c, estimate the cost to continue after visiting c
    # cont_cost[c] = softmin(d(c, depot), min_{j != c} d(c, j))
    
    # Get distances from each feasible node to all other feasible nodes
    d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)].copy()
    
    # Set diagonal to a large value (not inf, to avoid numerical issues with log-sum-exp)
    np.fill_diagonal(d_feas_feas, 1e9)
    
    # For each node i, compute softmin over all other feasible nodes
    # softmin(x_1, ..., x_n) = -1/beta * log(sum(exp(-beta * x_i)))
    # We want to combine d(c, depot) with softmin of d(c, j) for j != c
    
    beta = 2.0  # Temperature for soft-min
    
    # Soft-min of distances to other feasible nodes
    # d_feas_feas has shape (n_nodes, n_nodes), with diagonal = 1e9
    neg_scaled = -beta * d_feas_feas
    # Subtract max for numerical stability
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)
    sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
    log_sum_exp = np.log(sum_exp) + max_neg
    softmin_other = -log_sum_exp / beta
    
    # Now combine d_dep_feas with softmin_other using soft-min
    # We want softmin(d(c, depot), softmin_other(c))
    # This is equivalent to softmin over {d(c, depot), d(c, j1), d(c, j2), ...}
    # But we already computed softmin over other nodes, so we need to be careful
    
    # Actually, let's just do a direct soft-min over all options:
    # Option 1: go to depot: cost = d(c, depot)
    # Option 2: go to another feasible node j: cost = d(c, j)
    # The continuation cost is the soft-min of all these options
    
    # Create a matrix where row i has [d(feas[i], depot), d(feas[i], feas[0]), ..., d(feas[i], feas[n-1])]
    # with diagonal replaced by d(feas[i], depot) as a fallback? No, we want min of going to depot OR any other node
    
    # Let's construct the full set of options for each node
    # Option A: depot, cost = d_dep_feas[i]
    # Option B: node j (j != i), cost = d_feas_feas[i, j]
    
    # For soft-min, we combine all options
    # Stack: for each i, options are [d_dep_feas[i], d_feas_feas[i, 0], ..., d_feas_feas[i, n-1]]
    # But d_feas_feas[i, i] = 1e9, which effectively removes self
    
    # Build options matrix: shape (n_nodes, n_nodes + 1)
    options = np.empty((n_nodes, n_nodes + 1), dtype=np.float64)
    options[:, 0] = d_dep_feas  # Option to go to depot
    options[:, 1:] = d_feas_feas  # Options to go to other feasible nodes (diagonal is 1e9)
    
    # Soft-min over all options
    neg_scaled_opt = -beta * options
    max_neg_opt = np.max(neg_scaled_opt, axis=1, keepdims=True)
    exp_neg_opt = np.exp(neg_scaled_opt - max_neg_opt)
    sum_exp_opt = np.sum(exp_neg_opt, axis=1, keepdims=True)
    log_sum_exp_opt = np.log(sum_exp_opt) + max_neg_opt
    cont_cost = -log_sum_exp_opt / beta
    
    # Capacity-aware lookahead:
    # Penalize nodes that have high demand relative to remaining capacity after visiting them
    # This encourages filling routes more completely before closing
    remaining_after_visit = rest_capacity - demands[feas_nodes]
    cap_pressure = demands[feas_nodes] / (rest_capacity + 1e-9)
    
    # Regret term: penalize edges that are longer than the average
    mean_dist = np.mean(d_cur_feas)
    regret = mean_dist - d_cur_feas  # Favors shorter edges
    
    # Weighted combination:
    # score = savings - W1 * d_cur_feas + W2 * regret - W3 * cont_cost + W4 * (1 - cap_pressure)
    # Higher score is better
    
    W1 = 0.5   # Penalty on direct distance
    W2 = 1.0   # Regret weight
    W3 = 0.3   # Continuation cost weight
    W4 = 0.2   # Capacity pressure weight (encourages filling routes)
    
    scores = savings - W1 * d_cur_feas + W2 * regret - W3 * cont_cost + W4 * (1.0 - cap_pressure)
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
