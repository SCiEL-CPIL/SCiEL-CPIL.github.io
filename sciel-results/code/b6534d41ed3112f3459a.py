"""TSP constructive heuristic with adaptive 3-step lookahead and minimax bottleneck guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a hybrid of adaptive 3-step lookahead and minimax bottleneck.
    
    The primary score is the 3-step lookahead cost:
    d(current, c) + min_{b != c} ( d(c, b) + min_{a != b, c} ( d(b, a) + gamma * d(a, dest) ) )
    
    We also compute a minimax component: the maximum distance from candidate c to any other
    unvisited node. This prevents the tour from creating a very long edge to a distant city
    later. The final score is:
        total_score = lookahead_cost + lambda_minimax * max_dist_to_others
    
    lambda_minimax is small (0.05) so it acts as a tie-breaker / guard against extreme edges.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    # Check deadline early
    if _is_deadline_passed():
        # Fallback to pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Adaptive gamma
    gamma = max(0.0, 1.0 - (n_unvisited / 25.0))
    
    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # --- Minimax component: max distance from each candidate to any other unvisited node ---
    # For each i, max over j != i of submatrix[i, j]
    # Set diagonal to 0 so it doesn't affect max, then take max along axis 1
    submatrix_for_max = submatrix.copy()
    np.fill_diagonal(submatrix_for_max, 0.0)
    max_dists = np.max(submatrix_for_max, axis=1)
    
    # --- 3-step lookahead cost (vectorized where possible) ---
    # M_base[b, a] = d(b, a) + gamma * d(a, dest), with diagonal inf
    M_base = submatrix + gamma * dists_to_dest[np.newaxis, :]
    np.fill_diagonal(M_base, np.inf)
    
    best_total_score = np.inf
    best_candidate_idx = 0
    
    lambda_minimax = 0.05
    
    for i in range(n_unvisited):
        # Exclude a = i
        M_for_i = M_base.copy()
        M_for_i[:, i] = np.inf
        
        # inner_mins[b] = min_{a != i, a != b} (d(b, a) + gamma * d(a, dest))
        inner_mins = np.min(M_for_i, axis=1)
        
        # step1_costs[b] = d(c_i, b) + inner_mins[b], for b != i
        step1_costs = submatrix[i, :] + inner_mins
        step1_costs[i] = np.inf
        
        best_cont_cost = np.min(step1_costs)
        
        lookahead_cost = dists_from_current[i] + best_cont_cost
        minimax_cost = lambda_minimax * max_dists[i]
        
        total_score = lookahead_cost + minimax_cost
        
        if total_score < best_total_score:
            best_total_score = total_score
            best_candidate_idx = i
            
    return int(unvisited_nodes[best_candidate_idx])

# EVOLVE-BLOCK-END
