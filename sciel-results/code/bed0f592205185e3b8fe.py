"""CVRP heuristic: savings + geometric sweep + 2-step soft-min continuation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Clarke-Wright savings
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Geometric alignment: excess path length when going depot->current->feas vs direct depot->feas
    # Smaller excess = more aligned with sweep direction
    excess = d_cur_feas + d_dep_feas - d_cur_dep
    eps = 1e-10
    excess_range = np.max(excess) - np.min(excess) + eps
    align_norm = (excess - np.min(excess)) / excess_range  # 0 = best aligned
    
    # 2-step soft-min continuation cost
    d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
    np.fill_diagonal(d_feas_feas, np.inf)
    
    min_to_other = np.min(d_feas_feas, axis=1)
    
    # Soft-min of return-to-depot and min-to-other for 1-step continuation
    beta = 8.0
    a = -beta * d_dep_feas
    b = -beta * min_to_other
    m = np.maximum(a, b)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    logsumexp_val = m_safe + np.log(np.exp(a - m_safe) + np.exp(b - m_safe))
    cont_1step = -logsumexp_val / beta
    
    # 2-step: for each i, softmin over j != i of [d(i,j) + cont_1step[j]]
    M = d_feas_feas + cont_1step[np.newaxis, :]
    np.fill_diagonal(M, np.inf)
    
    beta2 = 6.0
    neg_M = -beta2 * M
    row_max = np.max(neg_M, axis=1)
    row_max_safe = np.where(np.isfinite(row_max), row_max, 0.0)
    logsumexp_rows = row_max_safe + np.log(np.sum(np.exp(neg_M - row_max_safe[:, np.newaxis]), axis=1))
    cont_cost = -logsumexp_rows / beta2
    
    # Normalize all terms
    savings_range = np.max(savings) - np.min(savings) + eps
    savings_norm = (savings - np.min(savings)) / savings_range
    
    dist_range = np.max(d_cur_feas) - np.min(d_cur_feas) + eps
    dist_norm = (d_cur_feas - np.min(d_cur_feas)) / dist_range
    
    cont_range = np.max(cont_cost) - np.min(cont_cost) + eps
    cont_norm = (cont_cost - np.min(cont_cost)) / cont_range
    
    # Combined score: prefer high savings, low distance, good alignment, low continuation
    W_sav = 1.3
    W_dist = 0.7
    W_align = 0.5
    W_cont = 0.5
    
    scores = W_sav * savings_norm - W_dist * dist_norm - W_align * align_norm - W_cont * cont_norm
    
    best_idx = np.argmax(scores)
    return int(feas_nodes[best_idx])
# EVOLVE-BLOCK-END
