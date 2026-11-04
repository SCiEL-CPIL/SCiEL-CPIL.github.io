# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas for permutation p.
    delta[i, j] = cost change if we swap positions i and j.
    """
    n = len(p)
    # Precompute
    # Let Bp = b[np.ix_(p, p)]
    Bp = b[np.ix_(p, p)]
    
    # We want delta[i, j] for i != j.
    # delta[i, j] = sum_{k != i, j} a[k, i] (Bp[k, j] - Bp[k, i]) + a[k, j] (Bp[k, i] - Bp[k, j])
    #             + a[i, k] (Bp[j, k] - Bp[i, k]) + a[j, k] (Bp[i, k] - Bp[j, k])
    #             + a[i, j] (Bp[j, i] - Bp[i, i]) + a[j, i] (Bp[i, j] - Bp[j, j])
    #             + a[i, i] (Bp[j, j] - Bp[i, i]) + a[j, j] (Bp[i, i] - Bp[j, j])
    # Note: Bp is symmetric if b is symmetric.
    
    # Let's use the vectorized formula:
    # For fixed i, j:
    # The term involving k (k != i, j) is:
    # (a[:, i] - a[:, j]) @ (Bp[:, j] - Bp[:, i])  -- but we must exclude i and j from the sum
    # Let's define:
    # v1 = a[:, i] - a[:, j]
    # v2 = Bp[:, j] - Bp[:, i]
    # Sum over k != i, j of v1[k] * v2[k]
    
    # This is O(n^3) if done naively for all i, j.
    # We can compute it as:
    # D = (a[:, i] - a[:, j]) @ (Bp[:, j] - Bp[:, i])
    # But we need to subtract the terms for k=i and k=j.
    
    # Let's try to compute the full matrix D where D[i, j] is the sum over ALL k.
    # D[i, j] = sum_k (a[k, i] - a[k, j]) * (Bp[k, j] - Bp[k, i])
    #         = sum_k a[k, i] Bp[k, j] - sum_k a[k, i] Bp[k, i] - sum_k a[k, j] Bp[k, j] + sum_k a[k, j] Bp[k, i]
    #         = (a[:, i] @ Bp[:, j]) - (a[:, i] @ Bp[:, i]) - (a[:, j] @ Bp[:, j]) + (a[:, j] @ Bp[:, i])
    
    # Let M1[i, j] = a[:, i] @ Bp[:, j]
    # Let M2[i, j] = a[:, i] @ Bp[:, i] (this is just a column of a matrix product? No, Bp[:, i] depends on i)
    # Actually, M2[i] = a[:, i] @ Bp[:, i] = sum_k a[k, i] Bp[k, i]
    # M3[i] = a[:, i] @ Bp[:, j] ... wait, the indices are mixed.
    
    # Let's compute:
    # S1 = a.T @ Bp  (n x n), where S1[i, j] = sum_k a[k, i] Bp[k, j]
    # Then D[i, j] = S1[i, j] - S1[i, i] - S1[j, j] + S1[j, i]
    
    S1 = a.T @ Bp
    # D_full[i, j] = S1[i, j] - S1[i, i] - S1[j, j] + S1[j, i]
    diag_S1 = np.diag(S1)
    D_full = S1 - diag_S1[:, None] - diag_S1[None, :] + S1.T
    
    # Now we need to subtract the terms where k=i and k=j from the sum.
    # The formula above sums over ALL k.
    # We need to remove the contributions from k=i and k=j.
    # Contribution of k=i to the sum for pair (i, j):
    # (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # Contribution of k=j to the sum for pair (i, j):
    # (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    
    # Let's compute these correction terms.
    # corr_i[i, j] = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # corr_j[i, j] = (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    
    # Note: a[i, j] is A[i, j]. Bp[i, j] is B[p[i], p[j]].
    
    # Let's build the correction matrix.
    # For each i, j:
    # term_i = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # term_j = (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    
    # This is still O(n^2) to compute explicitly.
    # Let's see if we can vectorize.
    
    # a_ii = a.diagonal()
    # a_jj = a.diagonal()
    # Bp_ii = Bp.diagonal()
    
    # term_i[i, j] = (a_ii[i] - a[i, j]) * (Bp[i, j] - Bp_ii[i])
    # term_j[i, j] = (a[j, i] - a_ii[j]) * (Bp_ii[j] - Bp[j, i])
    
    a_ii = np.diag(a)
    Bp_ii = np.diag(Bp)
    
    # term_i = (a_ii[:, None] - a) * (Bp - Bp_ii[:, None])
    term_i = (a_ii[:, None] - a) * (Bp - Bp_ii[:, None])
    
    # term_j[i, j] = (a[j, i] - a_ii[j]) * (Bp_ii[j] - Bp[j, i])
    # Let's transpose to match indices.
    # term_j[i, j] corresponds to row i, col j.
    # a[j, i] is a.T[i, j].
    # a_ii[j] is a_ii.T[i, j] (same as a_ii[i, j] if we index carefully? No, a_ii is a vector).
    # Let's use broadcasting.
    # a[j, i] -> a.T[i, j]
    # a_ii[j] -> a_ii.T[i, j] is wrong. a_ii[j] is the j-th element.
    # So we need a_ii[None, :] to broadcast along rows?
    # term_j[i, j] = (a.T[i, j] - a_ii[j]) * (Bp_ii[j] - Bp.T[i, j])
    
    term_j = (a.T - a_ii[None, :]) * (Bp_ii[None, :] - Bp.T)
    
    # The correction to subtract is term_i + term_j.
    # Wait, the sum in D_full includes k=i and k=j.
    # We want the sum over k != i, j.
    # So we subtract the terms for k=i and k=j.
    
    D_corrected = D_full - term_i - term_j
    
    # Now, is this the full delta?
    # Let's check the self/cross terms for the swap itself.
    # The cost change also involves the terms where the indices are i and j in the A matrix directly interacting with the swapped B positions.
    # Actually, the derivation above:
    # Cost = sum_{u, v} A[u, v] B[p[u], p[v]]
    # Swap i, j.
    # New Cost = Cost - (terms involving i or j in u or v) + (new terms involving i or j in u or v)
    # The terms involving i or j are:
    # u=i, v!=i: A[i, v] B[p[i], p[v]]
    # u!=i, v=i: A[u, i] B[p[u], p[i]]
    # u=i, v=i: A[i, i] B[p[i], p[i]]
    # Similarly for j.
    # And the cross terms u=i, v=j and u=j, v=i.
    
    # The formula D_full = S1 - ... derived from:
    # sum_k (A[k, i] - A[k, j]) (Bp[k, j] - Bp[k, i])
    # This accounts for the change in the "column" part (v=i or v=j) and "row" part (u=i or u=j) for all k.
    # Let's verify.
    # Delta = sum_{u,v} A[u,v] (B[p'[u], p'[v]] - B[p[u], p[v]])
    # Only terms where p'[u] != p[u] or p'[v] != p[v] change.
    # p'[i] = p[j], p'[j] = p[i].
    # So u=i or v=i or u=j or v=j.
    # 
    # Terms with v=i (and u!=i, j):
    # A[u, i] (B[p[u], p[i]] -> B[p[u], p[j]]) => A[u, i] (Bp[u, j] - Bp[u, i])
    # Terms with v=j (and u!=i, j):
    # A[u, j] (B[p[u], p[j]] -> B[p[u], p[i]]) => A[u, j] (Bp[u, i] - Bp[u, j])
    # Sum: sum_{u!=i,j} (A[u, i] - A[u, j]) (Bp[u, j] - Bp[u, i])
    # This matches the first part of D_full (with u as k).
    
    # Terms with u=i (and v!=i, j):
    # A[i, v] (B[p[i], p[v]] -> B[p[j], p[v]]) => A[i, v] (Bp[j, v] - Bp[i, v])
    # Terms with u=j (and v!=i, j):
    # A[j, v] (B[p[j], p[v]] -> B[p[i], p[v]]) => A[j, v] (Bp[i, v] - Bp[j, v])
    # Sum: sum_{v!=i,j} (A[i, v] - A[j, v]) (Bp[j, v] - Bp[i, v])
    # Note: Bp is symmetric, so Bp[j, v] = Bp[v, j].
    # This term is sum_{v!=i,j} (A[i, v] - A[j, v]) (Bp[v, j] - Bp[v, i])
    # This is the same as the first part but with A and Bp roles swapped in a sense?
    # Actually, since B is symmetric, Bp is symmetric.
    # The first part was sum_k (A[k, i] - A[k, j]) (Bp[k, j] - Bp[k, i]).
    # The second part is sum_k (A[i, k] - A[j, k]) (Bp[k, j] - Bp[k, i]).
    # These are different!
    
    # So D_full only captured the "column" part (v=i or v=j).
    # We need to add the "row" part (u=i or u=j).
    
    # Let's compute the row part similarly.
    # S2 = Bp.T @ a  (n x n), where S2[i, j] = sum_k Bp[k, i] A[k, j] = sum_k Bp[i, k] A[k, j]
    # Wait, Bp.T @ a -> [i, j] = sum_k Bp[k, i] a[k, j].
    # We want sum_k (A[i, k] - A[j, k]) (Bp[k, j] - Bp[k, i])
    # = sum_k A[i, k] Bp[k, j] - A[i, k] Bp[k, i] - A[j, k] Bp[k, j] + A[j, k] Bp[k, i]
    # = (a[i, :] @ Bp[:, j]) - (a[i, :] @ Bp[:, i]) - (a[j, :] @ Bp[:, j]) + (a[j, :] @ Bp[:, i])
    
    S2 = a @ Bp
    # S2[i, j] = sum_k a[i, k] Bp[k, j]
    # Row part full sum:
    # R_full[i, j] = S2[i, j] - S2[i, i] - S2[j, j] + S2[j, i]
    diag_S2 = np.diag(S2)
    R_full = S2 - diag_S2[:, None] - diag_S2[None, :] + S2.T
    
    # Subtract k=i and k=j terms for the row part.
    # term_i_row[i, j] = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])  -- Wait, indices are different.
    # For row part, the term for k=i is:
    # (A[i, i] - A[j, i]) * (Bp[i, j] - Bp[i, i]) ? No.
    # The sum is over k. The term for k=i is:
    # (A[i, i] - A[j, i]) * (Bp[i, j] - Bp[i, i])
    # The term for k=j is:
    # (A[i, j] - A[j, j]) * (Bp[j, j] - Bp[j, i])
    
    term_i_row = (a - a.T) * (Bp - Bp_ii[:, None]) # Check indices
    # a[i, k] - a[j, k]. For k=i: a[i, i] - a[j, i].
    # Bp[k, j] - Bp[k, i]. For k=i: Bp[i, j] - Bp[i, i].
    # So term_i_row[i, j] should be (a[i, i] - a[j, i]) * (Bp[i, j] - Bp[i, i]).
    # My formula: (a - a.T) * ...
    # (a - a.T)[i, j] = a[i, j] - a[j, i]. This is for general i, j.
    # We need the value at k=i for the sum over k.
    # This is getting confusing with indices.
    
    # Let's just compute the correction terms explicitly for the row part.
    # corr_row_i[i, j] = (a[i, i] - a[j, i]) * (Bp[i, j] - Bp[i, i])
    # corr_row_j[i, j] = (a[i, j] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    
    # a[i, i] is a_ii[i]. a[j, i] is a.T[i, j].
    # Bp[i, j] is Bp[i, j]. Bp[i, i] is Bp_ii[i].
    term_i_row = (a_ii[:, None] - a.T) * (Bp - Bp_ii[:, None])
    
    # a[i, j] is a[i, j]. a[j, j] is a_ii[j].
    # Bp[j, j] is Bp_ii[j]. Bp[j, i] is Bp.T[i, j].
    term_j_row = (a - a_ii[None, :]) * (Bp_ii[None, :] - Bp.T)
    
    R_corrected = R_full - term_i_row - term_j_row
    
    # Total delta = D_corrected + R_corrected
    # But we double counted the terms where BOTH u and v are in {i, j}?
    # The "column" part sum was over u (k), v fixed to i or j.
    # The "row" part sum was over v (k), u fixed to i or j.
    # The terms where u=i, v=i are in both?
    # Column part: v=i. u=i is included in sum over u!=i,j? No, we excluded u=i, j.
    # So the sums are over u != i, j and v != i, j.
    # So there is no overlap.
    
    # However, we missed the terms where u and v are exactly i and j (the cross terms).
    # The original cost has:
    # A[i, i] B[p[i], p[i]] + A[i, j] B[p[i], p[j]] + A[j, i] B[p[j], p[i]] + A[j, j] B[p[j], p[j]]
    # New cost has:
    # A[i, i] B[p[j], p[j]] + A[i, j] B[p[j], p[i]] + A[j, i] B[p[i], p[j]] + A[j, j] B[p[i], p[i]]
    
    # Delta_self = A[i, i] (Bp[j, j] - Bp[i, i]) + A[i, j] (Bp[j, i] - Bp[i, j])
    #             + A[j, i] (Bp[i, j] - Bp[j, i]) + A[j, j] (Bp[i, i] - Bp[j, j])
    
    # Let's add this to the delta matrix.
    # delta_self[i, j] = a[i, i] * (Bp[j, j] - Bp[i, i]) + a[i, j] * (Bp[j, i] - Bp[i, j])
    #                  + a[j, i] * (Bp[i, j] - Bp[j, i]) + a[j, j] * (Bp[i, i] - Bp[j, j])
    
    # a[i, i] (Bp_ii[j] - Bp_ii[i])
    # a[i, j] (Bp.T[i, j] - Bp[i, j])  (since Bp is symmetric, Bp[j, i] = Bp[i, j], so this is 0? No, Bp[j, i] is B[p[j], p[i]]. If B is symmetric, Bp is symmetric. So Bp[j, i] = Bp[i, j]. The term is 0.)
    # Wait, if B is symmetric, Bp is symmetric.
    # Bp[j, i] = Bp[i, j].
    # So A[i, j] (Bp[j, i] - Bp[i, j]) = 0.
    # Similarly A[j, i] (Bp[i, j] - Bp[j, i]) = 0.
    
    # So delta_self = a[i, i] (Bp_ii[j] - Bp_ii[i]) + a[j, j] (Bp_ii[i] - Bp_ii[j])
    #                = (a[i, i] - a[j, j]) (Bp_ii[j] - Bp_ii[i])
    
    delta_self = (a_ii[:, None] - a_ii[None, :]) * (Bp_ii[None, :] - Bp_ii[:, None])
    
    delta_matrix = D_corrected + R_corrected + delta_self
    
    # Set diagonal to 0
    np.fill_diagonal(delta_matrix, 0)
    
    return delta_matrix

def _local_search_fast(a, b, p, current_cost, max_iters=1000, time_limit=10.0):
    """
    Fast steepest-descent local search using precomputed delta matrix.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute delta matrix
        delta_matrix = _compute_delta_matrix(a, b, p)
        
        # Find the most negative delta
        min_idx = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
        i, j = min_idx
        min_delta = delta_matrix[i, j]
        
        if min_delta >= 0:
            break  # Local minimum
        
        # Perform swap
        p[i], p[j] = p[j], p[i]
        cost += min_delta
    
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Run simulated annealing for a given duration.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        # Compute delta for this specific swap
        # To avoid full matrix computation, we can compute just this delta.
        # But _compute_delta_matrix is O(n^2). Computing single delta is O(n).
        # Let's implement a fast single delta computation.
        
        pi = p[i]
        pj = p[j]
        
        # Using the logic from _compute_delta_matrix but for single i, j
        # D_part = sum_{k!=i,j} (a[k, i] - a[k, j]) (b[p[k], pj] - b[p[k], pi])
        # R_part = sum_{k!=i,j} (a[i, k] - a[j, k]) (b[pj, p[k]] - b[pi, p[k]])
        # Self = (a[i, i] - a[j, j]) (b[pj, pj] - b[pi, pi])
        
        k = np.arange(n)
        mask = (k != i) & (k != j)
        k_masked = k[mask]
        
        # D_part
        d_a = a[k_masked, i] - a[k_masked, j]
        d_b = b[p[k_masked], pj] - b[p[k_masked], pi]
        d_part = np.dot(d_a, d_b)
        
        # R_part
        r_a = a[i, k_masked] - a[j, k_masked]
        r_b = b[pj, p[k_masked]] - b[pi, p[k_masked]]
        r_part = np.dot(r_a, r_b)
        
        # Self
        self_part = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
        
        delta = int(d_part + r_part + self_part)
        
        new_cost = cost + delta
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    return best_p, best_cost

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    candidates = [(p_greedy, cost_greedy), (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    
    # Phase 2: Local search to polish initial solutions
    for idx in range(3):
        p_c, cost_c = candidates[idx]
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 30.0:
            break
        p_ls, cost_ls = _local_search_fast(a, b, p_c, cost_c, max_iters=100, time_limit=5.0)
        candidates[idx] = (p_ls, cost_ls)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Re-sort candidates
    candidates.sort(key=lambda x: x[1])
    
    # Phase 3: SA runs
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0  # Reserve 20s for final polish
    
    # Run 1
    seg1_time = remaining * 0.4
    T_start = 500.0
    T_end = 5.0
    p1, cost1 = _sa_run(a, b, candidates[0][0], candidates[0][1], seg1_time, T_start, T_end)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1.copy()
    
    # Run 2
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    if remaining > 10:
        seg2_time = remaining * 0.5
        T_start = 300.0
        T_end = 3.0
        p2, cost2 = _sa_run(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end)
        if cost2 < best_cost:
            best_cost = cost2
            best_p = p2.copy()
    
    # Run 3
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 20.0
    if remaining > 10:
        seg3_time = remaining * 0.6
        T_start = 200.0
        T_end = 2.0
        p3, cost3 = _sa_run(a, b, candidates[2][0], candidates[2][1], seg3_time, T_start, T_end)
        if cost3 < best_cost:
            best_cost = cost3
            best_p = p3.copy()
    
    # Phase 4: Final local search polish
    elapsed = time.time() - start_time
    polish_time = min(15.0, max(0.0, total_deadline - elapsed - 2.0))
    if polish_time > 0.5:
        best_p, best_cost = _local_search_fast(a, b, best_p, best_cost, max_iters=5000, time_limit=polish_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
