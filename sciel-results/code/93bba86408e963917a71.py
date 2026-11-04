"""
Tabu Search + Or-opt Solver for QAPLIB wil100.

Strategy:
1. Initialization: Greedy assignment based on row sums (centrality matching).
2. Optimization: 
   - Primary: Vectorized Best-Improvement Tabu Search.
     - Neighborhood: All pairs (i, j) of facilities.
     - Vectorization: Compute the cost delta for all n*(n-1)/2 swaps in O(n^3) 
       total per step (using matrix multiplications) instead of O(n^3) per step 
       with high constant factors. For n=100, this allows thousands of best-improvement 
       steps per second.
   - Secondary: Or-opt (block move) local search to break symmetry and escape 
     local minima that are invariant to pairwise swaps.
3. Tabu List: Prevent cycling by forbidding the reverse of recent moves.
"""

import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _init_greedy(a, b):
    """
    Initialize permutation by matching facilities with high flow row sums 
    to locations with high distance row sums.
    """
    n = len(a)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    fac_order = np.argsort(-a_sums)
    # Sort locations by descending distance sum
    loc_order = np.argsort(-b_sums)
    
    p = np.empty(n, dtype=np.int64)
    for i, f in enumerate(fac_order):
        p[f] = loc_order[i]
        
    return p

def _compute_all_deltas(a, b, p):
    """
    Compute the cost change for swapping any pair (i, j) in the current permutation p.
    Returns a matrix D of shape (n, n) where D[i, j] is the delta cost of swapping p[i] and p[j].
    
    Formula:
    Let u = p[i], v = p[j].
    Delta = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
            + sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
            + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
    
    We can vectorize this.
    Let A_diff = A - A.T (n x n)
    Let B_p = B[np.ix_(p, p)] (n x n)
    
    The cross terms can be written as:
    Term1[i, j] = sum_k (a[i,k] - a[j,k]) * (B_p[j, k] - B_p[i, k])  (Note: p[k] index in B_p is k)
                 = sum_k a[i,k]B_p[j,k] - a[i,k]B_p[i,k] - a[j,k]B_p[j,k] + a[j,k]B_p[i,k]
    
    This looks like matrix multiplications.
    Let M1 = A @ B_p.T  (M1[i, j] = sum_k a[i,k] B_p[j, k])
    Let M2 = A @ B_p    (M2[i, j] = sum_k a[i,k] B_p[i, k] -> wait, B_p[i,k] is row i of B_p)
    
    Actually, let's derive carefully.
    Cost(p) = sum_{i,j} a[i,j] B_p[i,j]
    
    Swap i, j in p. Let u=p[i], v=p[j].
    New B'_p is obtained by swapping rows i, j and cols i, j of B_p.
    
    Delta = Cost(B'_p) - Cost(B_p)
    
    This is equivalent to:
    Delta = sum_{k} (a[i,k] - a[j,k]) * (B_p[j, k] - B_p[i, k])  # Row i changes to row j, Row j to row i? No.
    
    Let's use the standard delta formula for QAP swap (i, j):
    d(i, j) = (a[i,j] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
             + sum_{k != i, j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
             + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
             
    Let B_p = B[np.ix_(p, p)].
    b[p[j], p[k]] is B_p[j, k].
    b[p[i], p[k]] is B_p[i, k].
    b[p[k], p[j]] is B_p[k, j].
    b[p[k], p[i]] is B_p[k, i].
    
    So:
    T1[i, j] = sum_{k != i, j} (a[i,k] - a[j,k]) * (B_p[j, k] - B_p[i, k])
    T2[i, j] = sum_{k != i, j} (a[k,i] - a[k,j]) * (B_p[k, j] - B_p[k, i])
    
    We can compute sums over all k and then subtract k=i and k=j.
    
    S1[i, j] = sum_k (a[i,k] - a[j,k]) * (B_p[j, k] - B_p[i, k])
             = sum_k a[i,k]B_p[j,k] - a[i,k]B_p[i,k] - a[j,k]B_p[j,k] + a[j,k]B_p[i,k]
             
    Let X = A @ B_p.T  => X[i, j] = sum_k A[i,k] B_p[j,k]
    Let Y = A @ B_p    => Y[i, j] = sum_k A[i,k] B_p[i,k] (Note: this is row-wise dot, not standard MM)
    
    Wait, Y[i, j] = sum_k A[i,k] B_p[i,k]. This is not a standard matrix product result for all i,j at once easily without broadcasting.
    However, Y is just the row-wise dot product of A and B_p.
    Y[i, :] = A[i, :] @ B_p[i, :].T? No.
    Y[i, j] = dot(A[i, :], B_p[i, :]). This is a vector of length n for each i.
    
    Let's restructure.
    T1[i, j] = X[i, j] - Y[i, j] - X[j, j] + Y[j, j] ? No.
    
    Let's check indices.
    X[i, j] = sum_k A[i,k] B_p[j,k]
    X[j, j] = sum_k A[j,k] B_p[j,k]
    
    Term - a[i,k] B_p[i,k]: Let Z[i, j] = sum_k A[i,k] B_p[i,k]. This is constant for a given i, independent of j.
    So Z[i, :] = Z[i, 0] * ones.
    
    So S1[i, j] = X[i, j] - Z[i, 0] - X[j, j] + Z[j, 0]
    
    Similarly for T2:
    T2[i, j] = sum_k (A[k,i] - A[k,j]) * (B_p[k,j] - B_p[k,i])
             = sum_k A[k,i]B_p[k,j] - A[k,i]B_p[k,i] - A[k,j]B_p[k,j] + A[k,j]B_p[k,i]
             
    Let W = A.T @ B_p => W[i, j] = sum_k A[k,i] B_p[k,j]
    Let V = A.T @ B_p.T => V[i, j] = sum_k A[k,i] B_p[k,i] ? No.
    
    V[i, j] should be sum_k A[k,i] B_p[k,i].
    This is dot(A[:, i], B_p[:, i]).
    
    So S2[i, j] = W[i, j] - V[i, 0] - W[j, j] + V[j, 0]
    
    Let's verify V.
    V[i, j] = sum_k A[k,i] B_p[k,i].
    This is the element-wise product sum of column i of A and column i of B_p.
    This value is independent of j. So V[i, :] is constant.
    
    So:
    S1[i, j] = X[i, j] - RowDotA_Bp[i] - X[j, j] + RowDotA_Bp[j]
    S2[i, j] = W[i, j] - ColDotA_Bp[i] - W[j, j] + ColDotA_Bp[j]
    
    Where:
    X = A @ B_p.T
    W = A.T @ B_p
    RowDotA_Bp[i] = sum_k A[i,k] B_p[i,k]
    ColDotA_Bp[i] = sum_k A[k,i] B_p[k,i]
    
    Then Delta[i, j] = (A[i,j] - A[j,i]) * (B_p[j,i] - B_p[i,j]) + S1[i,j] + S2[i,j]
    
    We must subtract the k=i and k=j terms from S1 and S2 because the sums above include them, but the formula for swap delta excludes them (they are handled by the direct term).
    
    Correction for S1 (sum over k != i, j):
    Full sum S1_full[i, j] includes k=i and k=j.
    k=i term in S1_full: (A[i,i]-A[j,i]) * (B_p[j,i] - B_p[i,i])
    k=j term in S1_full: (A[i,j]-A[j,j]) * (B_p[j,j] - B_p[i,j])
    
    Correction for S2 (sum over k != i, j):
    Full sum S2_full[i, j] includes k=i and k=j.
    k=i term in S2_full: (A[i,i]-A[i,j]) * (B_p[i,j] - B_p[i,i])
    k=j term in S2_full: (A[j,i]-A[j,j]) * (B_p[j,j] - B_p[j,i])
    
    So:
    Delta[i, j] = DirectTerm + (S1_full[i,j] - k_i_S1 - k_j_S1) + (S2_full[i,j] - k_i_S2 - k_j_S2)
    
    This can be computed for all i, j in O(n^3) using matrix multiplies and element-wise ops.
    """
    n = len(p)
    B_p = b[np.ix_(p, p)]
    
    # Matrix multiplications
    X = a @ B_p.T       # X[i, j] = sum_k a[i,k] B_p[j,k]
    W = a.T @ B_p       # W[i, j] = sum_k a[k,i] B_p[k,j]
    
    # Row-wise and Col-wise dot products
    # RowDotA_Bp[i] = sum_k a[i,k] B_p[i,k]
    RowDotA_Bp = np.sum(a * B_p, axis=1)
    # ColDotA_Bp[i] = sum_k a[k,i] B_p[k,i]
    ColDotA_Bp = np.sum(a * B_p, axis=0)
    
    # Construct S1 and S2 matrices
    # S1[i, j] = X[i, j] - RowDotA_Bp[i] - X[j, j] + RowDotA_Bp[j]
    X_diag = np.diag(X)
    S1 = X - RowDotA_Bp[:, None] - X_diag[None, :] + RowDotA_Bp[None, :]
    
    # S2[i, j] = W[i, j] - ColDotA_Bp[i] - W[j, j] + ColDotA_Bp[j]
    W_diag = np.diag(W)
    S2 = W - ColDotA_Bp[:, None] - W_diag[None, :] + ColDotA_Bp[None, :]
    
    # Direct term
    # D[i, j] = (a[i,j] - a[j,i]) * (B_p[j,i] - B_p[i,j])
    A_diff = a - a.T
    Bp_diff = B_p.T - B_p  # B_p[j,i] - B_p[i,j]
    Direct = A_diff * Bp_diff
    
    # Corrections for k=i and k=j
    # k=i in S1: (a[i,i]-a[j,i]) * (B_p[j,i] - B_p[i,i])
    # This depends on i and j.
    # Let's compute these correction matrices C1 and C2.
    
    # C1[i, j] corresponds to subtracting the k=i term from S1[i, j]
    # Term: (a[i,i] - a[j,i]) * (B_p[j,i] - B_p[i,i])
    # a[i,i] is diag_a. a[j,i] is a.T[i,j].
    # B_p[j,i] is B_p.T[i,j]. B_p[i,i] is diag_Bp.
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(B_p)
    
    # C1[i, j] = (diag_a[i] - a[j,i]) * (B_p[j,i] - diag_Bp[i])
    # Note: a[j,i] is a.T[i,j]
    # B_p[j,i] is B_p.T[i,j]
    
    C1 = (diag_a[:, None] - a.T) * (B_p.T - diag_Bp[:, None])
    
    # k=j in S1: (a[i,j]-a[j,j]) * (B_p[j,j] - B_p[i,j])
    # C1b[i, j] = (a[i,j] - diag_a[j]) * (diag_Bp[j] - B_p[i,j])
    C1b = (a - diag_a[None, :]) * (diag_Bp[None, :] - B_p)
    
    # C2[i, j] corresponds to subtracting k=i term from S2[i, j]
    # Term: (a[i,i]-a[i,j]) * (B_p[i,j] - B_p[i,i])
    # C2[i, j] = (diag_a[i] - a[i,j]) * (B_p[i,j] - diag_Bp[i])
    C2 = (diag_a[:, None] - a) * (B_p - diag_Bp[:, None])
    
    # k=j in S2: (a[j,i]-a[j,j]) * (B_p[j,j] - B_p[j,i])
    # C2b[i, j] = (a[j,i] - diag_a[j]) * (diag_Bp[j] - B_p[j,i])
    # a[j,i] is a.T[i,j]
    # B_p[j,i] is B_p.T[i,j]
    C2b = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - B_p.T)
    
    Deltas = Direct + (S1 - C1 - C1b) + (S2 - C2 - C2b)
    
    # Set diagonal to infinity (no self-swap)
    np.fill_diagonal(Deltas, np.inf)
    
    return Deltas

def _tabu_search(a, b, p, time_limit, tabu_len=20):
    """
    Best-Improvement Tabu Search.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    tabu_list = []  # List of (i, j) tuples, oldest first
    
    while time.time() - start_time < time_limit:
        # Compute all deltas
        deltas = _compute_all_deltas(a, b, p)
        
        # Find best move not in tabu list
        min_delta = np.inf
        best_i, best_j = -1, -1
        
        # Mask tabu moves
        # For efficiency, we can just check the top few candidates if n is large, 
        # but for n=100, checking all is fine if we vectorize the tabu check?
        # Tabu list is small (20). We can just loop through tabu list and mask.
        
        # Flatten deltas to 1D for easier argmin? Or just loop.
        # Let's find global min first.
        flat_idx = np.argmin(deltas)
        i, j = np.unravel_index(flat_idx, (n, n))
        
        # Check if tabu
        is_tabu = False
        for (ti, tj) in tabu_list:
            if (i == ti and j == tj) or (i == tj and j == ti):
                is_tabu = True
                break
        
        if is_tabu:
            # Find next best non-tabu
            # Mask tabu entries in deltas
            deltas_copy = deltas.copy()
            for (ti, tj) in tabu_list:
                deltas_copy[ti, tj] = np.inf
                deltas_copy[tj, ti] = np.inf
            flat_idx = np.argmin(deltas_copy)
            i, j = np.unravel_index(flat_idx, (n, n))
            min_delta = deltas_copy[i, j]
        else:
            min_delta = deltas[i, j]
            
        # Aspiration criterion: accept tabu if it improves best
        # (Simplified: just accept best non-tabu for now)
        
        if min_delta >= np.inf:
            break
            
        # Apply move
        p[i], p[j] = p[j], p[i]
        current_cost += int(min_delta)
        
        # Update tabu list
        tabu_list.append((i, j))
        if len(tabu_list) > tabu_len:
            tabu_list.pop(0)
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def _or_opt_move(a, b, p, start, length, new_start):
    """
    Move a block of facilities from [start, start+length) to new_start.
    Returns delta cost.
    """
    n = len(p)
    if new_start == start:
        return 0
    
    # Extract block
    block = p[start:start+length].copy()
    
    # Shift remaining
    if new_start < start:
        # Move block to earlier position
        # p[new_start : new_start+length] = block
        # p[new_start+length : start] = p[new_start : start-length]
        # p[start : start+length] = p[start-length : start]
        
        # This is complex to compute delta efficiently without full cost.
        # For Or-opt, we might just compute full cost if length is small?
        # Or derive delta.
        # Given time constraints, let's implement a simple version that computes cost diff.
        
        p_new = p.copy()
        # Shift right
        p_new[new_start+length : start] = p_new[new_start : start-length]
        p_new[new_start : new_start+length] = block
        # The rest is unchanged
        
        cost_old = _qap_cost(a, b, p)
        cost_new = _qap_cost(a, b, p_new)
        return cost_new - cost_old
    
    else:
        # Move block to later position
        # p[new_start : new_start+length] = block
        # p[start : new_start] = p[start+length : new_start+length]
        
        p_new = p.copy()
        p_new[start : new_start] = p_new[start+length : new_start+length]
        p_new[new_start : new_start+length] = block
        
        cost_old = _qap_cost(a, b, p)
        cost_new = _qap_cost(a, b, p_new)
        return cost_new - cost_old

def _or_opt_search(a, b, p, time_limit):
    """
    Local search using Or-opt moves (block moves of length 1, 2, 3).
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for length in [1, 2, 3]:
            for start in range(n - length + 1):
                for new_start in range(n - length + 1):
                    if new_start == start:
                        continue
                    # Avoid overlapping in a way that breaks permutation logic?
                    # The _or_opt_move handles the shift.
                    
                    delta = _or_opt_move(a, b, p, start, length, new_start)
                    if delta < 0:
                        # Apply move
                        block = p[start:start+length].copy()
                        if new_start < start:
                            p[new_start+length : start] = p[new_start : start-length]
                            p[new_start : new_start+length] = block
                        else:
                            p[start : new_start] = p[start+length : new_start+length]
                            p[new_start : new_start+length] = block
                            
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 145.0  # Safety margin
    start = time.time()
    
    np.random.seed(42)
    
    # 1. Initialization
    p = _init_greedy(a, b)
    
    # 2. Tabu Search (Main Engine)
    time_remaining = deadline - (time.time() - start)
    tabu_time = max(5.0, time_remaining * 0.6)
    p, cost = _tabu_search(a, b, p, tabu_time)
    
    # 3. Or-opt Polish
    time_remaining = deadline - (time.time() - start)
    or_time = max(5.0, time_remaining * 0.3)
    p, cost = _or_opt_search(a, b, p, or_time)
    
    # 4. Final Tabu Polish
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        p, cost = _tabu_search(a, b, p, time_remaining - 2)
        
    return p.tolist()

# EVOLVE-BLOCK-END
