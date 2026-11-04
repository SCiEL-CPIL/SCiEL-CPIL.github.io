import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_all_deltas(a, b, p):
    """
    Vectorized computation of all swap deltas for permutation p.
    Returns a 2D array of shape (n, n) where entry [i,j] is the delta for swapping i and j.
    The diagonal is 0. The matrix is symmetric.
    
    Delta(i, j) = Cost(p with i,j swapped) - Cost(p)
    """
    n = len(p)
    # We want to compute delta[i, j] for all i < j, then symmetrize.
    # Let's derive the vectorized formula.
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    # Swap i and j in p. Let p' be the new permutation.
    # p'[i] = p[j], p'[j] = p[i], p'[k] = p[k] for k != i,j.
    
    # The change in cost comes from terms where u or v is i or j.
    # Terms involving i or j:
    # 1. a[i,i] b[p[i], p[i]] -> a[i,i] b[p[j], p[j]]
    # 2. a[j,j] b[p[j], p[j]] -> a[j,j] b[p[i], p[i]]
    # 3. a[i,j] b[p[i], p[j]] -> a[i,j] b[p[j], p[i]]
    # 4. a[j,i] b[p[j], p[i]] -> a[j,i] b[p[i], p[j]]
    # 5. a[i,k] b[p[i], p[k]] -> a[i,k] b[p[j], p[k]] for k != i,j
    # 6. a[k,i] b[p[k], p[i]] -> a[k,i] b[p[k], p[j]] for k != i,j
    # 7. a[j,k] b[p[j], p[k]] -> a[j,k] b[p[i], p[k]] for k != i,j
    # 8. a[k,j] b[p[k], p[j]] -> a[k,j] b[p[k], p[i]] for k != i,j
    
    # Sum of changes:
    # Delta = (a[i,i](b[jj]-b[ii]) + a[j,j](b[ii]-b[jj]) + a[i,j](b[ji]-b[ij]) + a[j,i](b[ij]-b[ji]))
    #       + sum_{k!=i,j} (a[i,k](b[jk]-b[ik]) + a[k,i](b[kj]-b[ki]) + a[j,k](b[ik]-b[jk]) + a[k,j](b[ki]-b[kj]))
    
    # Let's use the formula:
    # Delta(i,j) = sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]]) 
    #             + sum_k (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    #             - (a[i,j] - a[j,j]) * (b[p[j], p[i]] - b[p[i], p[i]])  # Correction for k=j in first sum? No, let's stick to the direct expansion.
    
    # Actually, the standard vectorized delta formula is:
    # Let A_i = a[:, i], A_j = a[:, j]
    # Let B_i = b[p[i], :], B_j = b[p[j], :]
    # This is getting complicated to derive from scratch in the head. Let's use the property:
    # Cost(p) = trace(A P B P^T) ? No, P is a permutation matrix.
    # Cost = sum a[i,j] b[p[i], p[j]]
    
    # Let's use the pre-computed vectors.
    # For a fixed swap (i, j), the delta is:
    # sum_{k} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]]) 
    # + sum_{k} (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    # - (a[i,i] - a[j,j]) * (b[p[j], p[i]] - b[p[i], p[i]]) # Wait, k=i and k=j are included in the sums above.
    # Let's check the k=i term in the first sum:
    # (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])
    # Let's check the k=j term in the first sum:
    # (a[i,j] - a[j,j]) * (b[p[j], p[j]] - b[p[i], p[j]])
    
    # The formula I will use is:
    # D[i, j] = sum_{k} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]]) 
    #         + sum_{k} (a[k,i] - a[k,j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    # This double counts or misses diagonal terms?
    # Let's verify with n=2.
    # a = [[a11, a12], [a21, a22]], b = [[b11, b12], [b21, b22]]
    # p = [0, 1]. Cost = a11 b00 + a12 b01 + a21 b10 + a22 b11
    # Swap 0,1 -> p' = [1, 0]. Cost' = a11 b11 + a12 b10 + a21 b01 + a22 b00
    # Delta = Cost' - Cost
    # = a11(b11-b00) + a12(b10-b01) + a21(b01-b10) + a22(b00-b11)
    
    # Using formula for i=0, j=1:
    # Term 1: sum_k (a[0,k] - a[1,k]) * (b[p[1], p[k]] - b[p[0], p[k]])
    # k=0: (a00-a10) * (b[1,0] - b[0,0])
    # k=1: (a01-a11) * (b[1,1] - b[0,1])
    # Sum 1 = a00 b10 - a00 b00 - a10 b10 + a10 b00 + a01 b11 - a01 b01 - a11 b11 + a11 b01
    
    # Term 2: sum_k (a[k,0] - a[k,1]) * (b[p[k], p[1]] - b[p[k], p[0]])
    # k=0: (a00-a01) * (b[0,1] - b[0,0])
    # k=1: (a10-a11) * (b[1,1] - b[1,0])
    # Sum 2 = a00 b01 - a00 b00 - a01 b01 + a01 b00 + a10 b11 - a10 b10 - a11 b11 + a11 b10
    
    # Total = Sum 1 + Sum 2
    # = a00 b10 - 2 a00 b00 + a00 b01 - a10 b10 + a10 b00 + a01 b11 - 2 a01 b01 + a01 b00 + a10 b11 - 2 a10 b10 ... wait.
    
    # This manual derivation is error-prone. I will use the known efficient vectorized delta formula.
    # Let V_i = a[:, i] - a[i, :]  (Column i - Row i)
    # Let W_i = b[:, p[i]] - b[p[i], :] (Column p[i] - Row p[i])
    # This is not quite right either.
    
    # Correct Vectorized Delta:
    # Let's precompute:
    # A_col_diff[i, k] = a[i, k] - a[k, i]  (This is a[i,k] - a[k,i])
    # This is not what we need.
    
    # Let's use the formula:
    # Delta(i, j) = sum_{k} (a[i, k] - a[j, k]) * (b[p[j], p[k]] - b[p[i], p[k]]) 
    #             + sum_{k} (a[k, i] - a[k, j]) * (b[p[k], p[j]] - b[p[k], p[i]])
    #             - (a[i, i] - a[j, j]) * (b[p[j], p[i]] - b[p[i], p[i]])  # This correction is for the k=i term in the second sum and k=j in the first?
    
    # Actually, the simplest robust way is:
    # 1. Compute cost of p.
    # 2. For each i, compute the vector of costs if we swap i with any j.
    
    # Let's try a different vectorization strategy.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let's define M[i, j] = b[p[i], p[j]].
    # Cost = sum a[i,j] M[i,j].
    
    # If we swap p[i] and p[j], M changes only in rows i, j and cols i, j.
    # M_new[i, i] = b[p[j], p[j]]
    # M_new[i, j] = b[p[j], p[i]]
    # M_new[j, i] = b[p[i], p[j]]
    # M_new[j, j] = b[p[i], p[i]]
    # M_new[i, k] = b[p[j], p[k]] for k != i, j
    # M_new[k, i] = b[p[k], p[j]] for k != i, j
    # M_new[j, k] = b[p[i], p[k]] for k != i, j
    # M_new[k, j] = b[p[k], p[i]] for k != i, j
    
    # Delta = a[i,i](M_new[i,i]-M[i,i]) + a[i,j](M_new[i,j]-M[i,j]) + a[j,i](M_new[j,i]-M[j,i]) + a[j,j](M_new[j,j]-M[j,j])
    #       + sum_{k!=i,j} (a[i,k](M_new[i,k]-M[i,k]) + a[k,i](M_new[k,i]-M[k,i]) + a[j,k](M_new[j,k]-M[j,k]) + a[k,j](M_new[k,j]-M[k,j]))
    
    # We can compute this for all i,j pairs simultaneously using broadcasting.
    
    n = len(p)
    # Precompute b_p = b[np.ix_(p, p)]
    b_p = b[np.ix_(p, p)]
    
    # We want Delta[i, j].
    # Term 1: a[i,i] (b[p[j],p[j]] - b[p[i],p[i]])
    # Term 2: a[i,j] (b[p[j],p[i]] - b[p[i],p[j]])
    # Term 3: a[j,i] (b[p[i],p[j]] - b[p[j],p[i]])
    # Term 4: a[j,j] (b[p[i],p[i]] - b[p[j],p[j]])
    
    # Sum 1: sum_{k!=i,j} a[i,k] (b[p[j],p[k]] - b[p[i],p[k]])
    # Sum 2: sum_{k!=i,j} a[k,i] (b[p[k],p[j]] - b[p[k],p[i]])
    # Sum 3: sum_{k!=i,j} a[j,k] (b[p[i],p[k]] - b[p[j],p[k]])
    # Sum 4: sum_{k!=i,j} a[k,j] (b[p[k],p[i]] - b[p[k],p[j]])
    
    # Let's compute the sums including k=i and k=j, then subtract the corrections.
    
    # S1_full[i, j] = sum_k a[i,k] (b[p[j],p[k]] - b[p[i],p[k]])
    # This can be computed as:
    # sum_k a[i,k] b[p[j],p[k]] - sum_k a[i,k] b[p[i],p[k]]
    # Let B_p_col[k] = b[:, p[k]]? No.
    # Let's use matrix multiplication.
    # Let A = a.
    # Let B_perm = b[np.ix_(p, p)]
    # sum_k a[i,k] B_perm[j,k] = (A @ B_perm.T)[i, j]
    # sum_k a[i,k] B_perm[i,k] = (A @ B_perm.T)[i, i]
    
    # So S1_full[i, j] = (A @ B_perm.T)[i, j] - (A @ B_perm.T)[i, i]
    
    # Similarly for S3_full[i, j] (swapping i and j, so j becomes i):
    # S3_full[i, j] = sum_k a[j,k] (B_perm[i,k] - B_perm[j,k])
    # = (A @ B_perm.T)[j, i] - (A @ B_perm.T)[j, j]
    
    # Now for the column sums (S2 and S4).
    # S2_full[i, j] = sum_k a[k,i] (B_perm[k,j] - B_perm[k,i])
    # = sum_k a[k,i] B_perm[k,j] - sum_k a[k,i] B_perm[k,i]
    # = (A.T @ B_perm)[i, j] - (A.T @ B_perm)[i, i]
    
    # S4_full[i, j] = sum_k a[k,j] (B_perm[k,i] - B_perm[k,j])
    # = (A.T @ B_perm)[j, i] - (A.T @ B_perm)[j, j]
    
    # Now we have the full sums. We need to subtract the terms where k=i and k=j.
    # For S1_full[i, j]:
    # k=i term: a[i,i] (B_perm[j,i] - B_perm[i,i])
    # k=j term: a[i,j] (B_perm[j,j] - B_perm[i,j])
    # So S1[i, j] = S1_full[i, j] - a[i,i]*(B_perm[j,i] - B_perm[i,i]) - a[i,j]*(B_perm[j,j] - B_perm[i,j])
    
    # For S3_full[i, j]:
    # k=i term: a[j,i] (B_perm[i,i] - B_perm[j,i])
    # k=j term: a[j,j] (B_perm[i,j] - B_perm[j,j])
    # So S3[i, j] = S3_full[i, j] - a[j,i]*(B_perm[i,i] - B_perm[j,i]) - a[j,j]*(B_perm[i,j] - B_perm[j,j])
    
    # For S2_full[i, j]:
    # k=i term: a[i,i] (B_perm[i,j] - B_perm[i,i])
    # k=j term: a[j,i] (B_perm[j,j] - B_perm[j,i])
    # So S2[i, j] = S2_full[i, j] - a[i,i]*(B_perm[i,j] - B_perm[i,i]) - a[j,i]*(B_perm[j,j] - B_perm[j,i])
    
    # For S4_full[i, j]:
    # k=i term: a[i,j] (B_perm[i,i] - B_perm[i,j])
    # k=j term: a[j,j] (B_perm[j,i] - B_perm[j,j])
    # So S4[i, j] = S4_full[i, j] - a[i,j]*(B_perm[i,i] - B_perm[i,j]) - a[j,j]*(B_perm[j,i] - B_perm[j,j])
    
    # The diagonal terms (Term 1-4) are:
    # T1 = a[i,i] (B_perm[j,j] - B_perm[i,i])
    # T2 = a[i,j] (B_perm[j,i] - B_perm[i,j])
    # T3 = a[j,i] (B_perm[i,j] - B_perm[j,i])
    # T4 = a[j,j] (B_perm[i,i] - B_perm[j,j])
    
    # Total Delta[i, j] = S1[i,j] + S2[i,j] + S3[i,j] + S4[i,j] + T1 + T2 + T3 + T4
    
    # Let's assemble this vectorized.
    ABT = a @ b_p.T
    ATB = a.T @ b_p
    
    S1_full = ABT - ABT[:, np.newaxis] * 0 + ABT[np.arange(n), np.newaxis] # No, ABT[i,i] is scalar per row.
    # ABT is (n, n). ABT[i, i] is the diagonal.
    # We want S1_full[i, j] = ABT[i, j] - ABT[i, i]
    # This is ABT - np.diag(ABT)[:, np.newaxis]
    
    S1_full = ABT - np.diag(ABT)[:, np.newaxis]
    S3_full = ABT.T - np.diag(ABT)[:, np.newaxis].T # ABT[j, i] - ABT[j, j]
    
    S2_full = ATB - np.diag(ATB)[:, np.newaxis]
    S4_full = ATB.T - np.diag(ATB)[:, np.newaxis].T
    
    # Now corrections.
    # We need B_perm[i, i], B_perm[i, j], etc.
    # B_perm is (n, n).
    
    # a[i,i] is np.diag(a)
    diag_a = np.diag(a)
    
    # T1: diag_a[i] * (B_perm[j,j] - B_perm[i,i])
    # Vectorized: diag_a[:, np.newaxis] * (np.diag(b_p)[np.newaxis, :] - np.diag(b_p)[:, np.newaxis])
    T1 = diag_a[:, np.newaxis] * (np.diag(b_p)[np.newaxis, :] - np.diag(b_p)[:, np.newaxis])
    
    # T4: diag_a[j] * (B_perm[i,i] - B_perm[j,j])
    # This is just -T1.T? No, T1[i,j] uses a[i,i]. T4[i,j] uses a[j,j].
    T4 = diag_a[np.newaxis, :] * (np.diag(b_p)[:, np.newaxis] - np.diag(b_p)[np.newaxis, :])
    
    # T2: a[i,j] * (B_perm[j,i] - B_perm[i,j])
    T2 = a * (b_p.T - b_p)
    
    # T3: a[j,i] * (B_perm[i,j] - B_perm[j,i])
    T3 = a.T * (b_p - b_p.T)
    
    # Corrections for S1:
    # C1_ii = a[i,i] * (B_perm[j,i] - B_perm[i,i])
    # C1_ij = a[i,j] * (B_perm[j,j] - B_perm[i,j])
    C1_ii = diag_a[:, np.newaxis] * (b_p.T - np.diag(b_p)[:, np.newaxis])
    C1_ij = a * (np.diag(b_p)[np.newaxis, :] - b_p)
    S1 = S1_full - C1_ii - C1_ij
    
    # Corrections for S3:
    # C3_ii = a[j,i] * (B_perm[i,i] - B_perm[j,i])
    # C3_jj = a[j,j] * (B_perm[i,j] - B_perm[j,j])
    C3_ii = a.T * (np.diag(b_p)[:, np.newaxis] - b_p.T)
    C3_jj = diag_a[np.newaxis, :] * (b_p - np.diag(b_p)[np.newaxis, :])
    S3 = S3_full - C3_ii - C3_jj
    
    # Corrections for S2:
    # C2_ii = a[i,i] * (B_perm[i,j] - B_perm[i,i])
    # C2_ji = a[j,i] * (B_perm[j,j] - B_perm[j,i])
    C2_ii = diag_a[:, np.newaxis] * (b_p - np.diag(b_p)[:, np.newaxis])
    C2_ji = a.T * (np.diag(b_p)[np.newaxis, :] - b_p.T)
    S2 = S2_full - C2_ii - C2_ji
    
    # Corrections for S4:
    # C4_ij = a[i,j] * (B_perm[i,i] - B_perm[i,j])
    # C4_jj = a[j,j] * (B_perm[j,i] - B_perm[j,j])
    C4_ij = a * (np.diag(b_p)[:, np.newaxis] - b_p)
    C4_jj = diag_a[np.newaxis, :] * (b_p.T - np.diag(b_p)[np.newaxis, :])
    S4 = S4_full - C4_ij - C4_jj
    
    deltas = S1 + S2 + S3 + S4 + T1 + T2 + T3 + T4
    
    # Set diagonal to 0
    np.fill_diagonal(deltas, 0)
    # Ensure symmetry (numerical errors might break it slightly, but integer math should be exact)
    # deltas = (deltas + deltas.T) / 2 
    # Since inputs are integers, deltas should be integers.
    return deltas.astype(np.int64)

def _local_search(a, b, p, current_cost, time_limit=10.0):
    """
    Steepest-descent local search using vectorized deltas.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    while time.time() - start_time < time_limit:
        deltas = _compute_all_deltas(a, b, p)
        # Find the most negative delta
        min_delta = np.min(deltas)
        if min_delta >= 0:
            break
        
        # Find the indices
        i, j = np.unravel_index(np.argmin(deltas), deltas.shape)
        
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
        # Using the vectorized function is overkill for a single swap if we do many, 
        # but it's the only one we have. 
        # Alternatively, we can compute delta(i,j) from the full matrix if we update it incrementally, 
        # but that's complex. 
        # Let's just use the single swap delta function which is faster for 1 swap.
        
        # Re-implementing single swap delta for speed in SA loop
        pi = p[i]
        pj = p[j]
        
        t1 = np.dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
        t1 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
        t1 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
        
        t2 = np.dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
        t2 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
        t2 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
        
        t3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
        
        delta = int(t1 + t2 + t3)
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

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Initial Solutions
    # Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # Random
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = min(cost_greedy, cost_random)
    
    # Phase 2: Local Search Polish
    elapsed = time.time() - start_time
    ls_time = min(5.0, total_deadline - elapsed - 10.0) # Reserve 10s for SA and final LS
    if ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=ls_time)
        
    # Phase 3: Multi-start SA
    elapsed = time.time() - start_time
    sa_time_total = total_deadline - elapsed - 5.0 # Reserve 5s for final LS
    num_sa_runs = 3
    sa_time_each = sa_time_total / num_sa_runs
    
    for k in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 5.0:
            break
            
        # Diversify start
        if k == 0:
            p_start = best_p.copy()
            c_start = best_cost
        else:
            p_start = np.random.permutation(n).astype(np.int64)
            c_start = _qap_cost(a, b, p_start)
            # Quick LS to get to a local minimum
            p_start, c_start = _local_search(a, b, p_start, c_start, time_limit=1.0)
        
        T_start = 500.0 * (0.5 ** k) # Cool down for subsequent runs? Or keep same?
        T_end = 5.0
        p_sa, c_sa = _sa_run(a, b, p_start, c_start, sa_time_each, T_start, T_end)
        
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
            
    # Phase 4: Final Polish
    elapsed = time.time() - start_time
    final_ls_time = max(0.0, total_deadline - elapsed - 1.0)
    if final_ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=final_ls_time)
        
    return best_p.tolist()
