"""Traveling salesman constructive heuristic with Or-opt segment-based scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using Or-opt segment-based scoring.
    
    For each candidate, estimate the marginal cost of inserting a short
    segment (1, 2, or 3 nodes starting at the candidate) into the tour,
    then pick the candidate with the lowest estimated marginal cost.
    
    This is structurally different from per-node weighted lookahead:
    it evaluates multi-node patterns rather than independent node scores.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to pure nearest neighbor
    if time.time() - _start_time > TIME_LIMIT:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    dm = distance_matrix
    
    # --- 1-node score: just d(current, cand) + d(cand, dest) ---
    score1 = dm[current_node, unvisited_nodes] + dm[unvisited_nodes, destination_node]
    
    # --- 2-node score: for each candidate i, find best partner j ---
    # cost = d(cur, i) + d(i, j) + d(j, dest)
    # For each i, the best j minimizes d(i,j) + d(j,dest)
    # Vectorized: build n x n matrix of d(i,j) + d(j,dest)
    if n_unvisited >= 2:
        # d(i, j) for all pairs
        d_ij = dm[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
        # d(j, dest) for each j (column vector)
        d_j_dest = dm[unvisited_nodes, destination_node]
        # For each row i, compute d(i,j) + d(j,dest) over all j != i
        pair_cost = d_ij + d_j_dest[np.newaxis, :]
        np.fill_diagonal(pair_cost, np.inf)
        best_pair = np.min(pair_cost, axis=1)  # shape (n,)
        score2 = dm[current_node, unvisited_nodes] + best_pair
    else:
        score2 = score1.copy()
    
    # --- 3-node score: for each candidate i, find best (j, k) ---
    # cost = d(cur, i) + d(i, j) + d(j, k) + d(k, dest)
    # For each i, minimize over j != i, k != i, j: d(i,j) + d(j,k) + d(k,dest)
    if n_unvisited >= 3:
        # d(i, j) for all pairs
        d_ij = dm[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
        # d(j, k) for all pairs
        d_jk = dm[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
        # d(k, dest) for each k
        d_k_dest = dm[unvisited_nodes, destination_node]
        
        # For each (i, j) pair with i != j, find best k != i, j:
        # triple_cost[i, j] = d(i,j) + min_{k != i, j} (d(j,k) + d(k,dest))
        # d(j,k) + d(k,dest): shape (n, n) where row=j, col=k
        d_jk_plus_dest = d_jk + d_k_dest[np.newaxis, :]
        # For each j, we need min over k != j (and later k != i too)
        # First mask k == j
        d_jk_plus_dest_masked = d_jk_plus_dest.copy()
        np.fill_diagonal(d_jk_plus_dest_masked, np.inf)
        # best_k_for_j[j] = min_k (d(j,k) + d(k,dest)) with k != j
        best_k_for_j = np.min(d_jk_plus_dest_masked, axis=1)  # shape (n,)
        
        # Now for each (i, j) with i != j, triple_cost = d(i,j) + best_k_for_j[j]
        # But we also need k != i. If best_k_for_j[j] == i, we need second-best.
        # Approximation: use best_k_for_j[j] (ignoring the k != i constraint)
        # This is a slight relaxation but keeps it vectorized.
        triple_cost = d_ij + best_k_for_j[np.newaxis, :]
        np.fill_diagonal(triple_cost, np.inf)  # exclude i == j
        best_triple = np.min(triple_cost, axis=1)  # shape (n,)
        score3 = dm[current_node, unvisited_nodes] + best_triple
    else:
        score3 = score1.copy()
    
    # --- Combine scores with adaptive weighting ---
    # Early in construction (many unvisited), longer segments matter more
    # Late in construction, shorter segments are more reliable
    frac_remaining = n_unvisited / 50.0
    
    if n_unvisited >= 3:
        w1 = 0.2
        w2 = 0.3
        w3 = 0.5
    elif n_unvisited >= 2:
        w1 = 0.4
        w2 = 0.6
        w3 = 0.0
    else:
        w1 = 1.0
        w2 = 0.0
        w3 = 0.0
    
    # Normalize scores to [0,1] range for fair combination
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-10:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    combined = w1 * normalize(score1) + w2 * normalize(score2) + w3 * normalize(score3)
    
    best_idx = np.argmin(combined)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
