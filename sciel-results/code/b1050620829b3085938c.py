"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit (task says 120s, but let's use 110 for safety)

    # Precompute row sums for initial heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Initial rank-matching: heavy flows to central locations
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def compute_cost(p):
        """Compute exact QAP cost."""
        return np.sum(a * b[np.ix_(p, p)])

    best_cost = compute_cost(p)
    best_p = p.copy()

    # 2-opt local search with efficient delta computation
    def two_opt_search(perm, current_cost, max_time):
        n_local = len(perm)
        improved = True
        # Precompute some things that might help? 
        # Actually, the O(n) loop inside the O(n^2) loop is O(n^3).
        # For n=256, 256^3 = 16.7M operations. In Python this is too slow for a full pass.
        # We need a vectorized delta or a faster method.
        
        # Let's try to vectorize the delta calculation for a fixed i, varying j.
        # Delta(i, j) for swapping p[i] and p[j]:
        # Let u = p[i], v = p[j].
        # Old terms involving i or j:
        # T_old = sum_{k != i,j} [a[i,k]*b[u,p[k]] + a[k,i]*b[p[k],u] + a[j,k]*b[v,p[k]] + a[k,j]*b[p[k],v]]
        #       + a[i,j]*b[u,v] + a[j,i]*b[v,u]
        
        # New terms (swap u and v):
        # T_new = sum_{k != i,j} [a[i,k]*b[v,p[k]] + a[k,i]*b[p[k],v] + a[j,k]*b[u,p[k]] + a[k,j]*b[p[k],u]]
        #       + a[i,j]*b[v,u] + a[j,i]*b[u,v]
        
        # Delta = T_new - T_old
        # Delta = sum_{k != i,j} [ a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[k,i]*(b[p[k],v]-b[p[k],u]) 
        #                         + a[j,k]*(b[u,p[k]]-b[v,p[k]]) + a[k,j]*(b[p[k],u]-b[p[k],v]) ]
        #       + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        
        # Note: b[v,u]-b[u,v] = -(b[u,v]-b[v,u]).
        # So last two terms: (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        
        # Let's define vectors for fixed i (so u is fixed):
        # For each k, we need b[v, p[k]] - b[u, p[k]]. This is (b[v, :] - b[u, :])[p[k]].
        # Let diff_row = b[v, :] - b[u, :]  (shape n)
        # Let diff_col = b[:, v] - b[:, u]  (shape n)
        # Then for a specific k, the terms are:
        # a[i,k] * diff_row[p[k]] + a[k,i] * diff_col[p[k]]
        # + a[j,k] * (-diff_row[p[k]]) + a[k,j] * (-diff_col[p[k]])
        # = (a[i,k] - a[j,k]) * diff_row[p[k]] + (a[k,i] - a[k,j]) * diff_col[p[k])
        
        # So for fixed i and u=p[i], we can precompute:
        # For all j != i, we need to compute:
        # Delta(i, j) = sum_{k != i,j} [ (a[i,k] - a[j,k]) * diff_row[p[k]] + (a[k,i] - a[k,j]) * diff_col[p[k]] ]
        #                + (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])  where v=p[j]
        
        # This is still O(n) per j if we do it naively.
        # Can we vectorize over j?
        # Let's expand the sum:
        # Sum_{k != i,j} ...
        # = Sum_{k} ... - term_k=i - term_k=j
        
        # Let's define for fixed i, u:
        # V_row[k] = diff_row[p[k]]  (shape n)
        # V_col[k] = diff_col[p[k]]  (shape n)
        
        # Term for k in the sum:
        # T_k(j) = (a[i,k] - a[j,k]) * V_row[k] + (a[k,i] - a[k,j]) * V_col[k]
        #        = a[i,k]*V_row[k] - a[j,k]*V_row[k] + a[k,i]*V_col[k] - a[k,j]*V_col[k]
        #        = (a[i,k]*V_row[k] + a[k,i]*V_col[k]) - (a[j,k]*V_row[k] + a[k,j]*V_col[k])
        
        # Let C_i[k] = a[i,k]*V_row[k] + a[k,i]*V_col[k]
        # Let C_j[k] = a[j,k]*V_row[k] + a[k,j]*V_col[k]
        
        # Sum_{k != i,j} T_k(j) = Sum_{k != i,j} C_i[k] - Sum_{k != i,j} C_j[k]
        # = (Sum_{k} C_i[k] - C_i[i] - C_i[j]) - (Sum_{k} C_j[k] - C_j[i] - C_j[j])
        # = (Sum_C_i - C_i[i] - C_i[j]) - (Sum_C_j - C_j[i] - C_j[j])
        # = Sum_C_i - Sum_C_j - C_i[i] + C_j[i] - C_i[j] + C_j[j]
        
        # Here Sum_C_i = sum_k C_i[k] is constant for fixed i.
        # Sum_C_j = sum_k C_j[k] depends on j.
        # C_i[i], C_i[j] depend on i, j.
        # C_j[i], C_j[j] depend on i, j.
        
        # So Delta(i, j) = (Sum_C_i - Sum_C_j - C_i[i] + C_j[i] - C_i[j] + C_j[j])
        #                  + (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        
        # We can precompute for fixed i:
        # 1. V_row, V_col
        # 2. C_i vector (shape n)
        # 3. Sum_C_i
        # 4. C_i[i]
        
        # Then for each j, we need:
        # - Sum_C_j = sum_k (a[j,k]*V_row[k] + a[k,j]*V_col[k])
        #   This is dot(a[j, :], V_row) + dot(a[:, j], V_col)
        # - C_i[j] = a[i,j]*V_row[j] + a[j,i]*V_col[j]
        # - C_j[i] = a[j,i]*V_row[i] + a[i,j]*V_col[i]
        # - C_j[j] = a[j,j]*V_row[j] + a[j,j]*V_col[j]
        # - b[v,u] - b[u,v] where v=p[j], u=p[i]
        
        # So for fixed i, we can compute Delta(i, j) for all j in O(n^2) total?
        # Sum_C_j for all j:
        # Sum_C_j = a[j, :] @ V_row + a[:, j] @ V_col
        # This is (a @ V_row)[j] + (a.T @ V_col)[j]
        # So we can compute all Sum_C_j in O(n^2) using matrix-vector products.
        
        # Let's verify this logic.
        # V_row[k] = (b[v, :] - b[u, :])[p[k]]
        # V_col[k] = (b[:, v] - b[:, u])[p[k]]
        
        # C_i[k] = a[i,k]*V_row[k] + a[k,i]*V_col[k]
        # Sum_C_i = sum(C_i)
        
        # C_j[k] = a[j,k]*V_row[k] + a[k,j]*V_col[k]
        # Sum_C_j = sum(C_j) = a[j, :] @ V_row + a[:, j] @ V_col
        
        # This looks correct and much faster! O(n^2) per i, so O(n^3) total.
        # But wait, we are in a loop over i. If we do this for every i, it's O(n^3).
        # The previous method was O(n^3) with a large constant. This one has a smaller constant.
        
        # Let's implement this.
        
        while time.time() < max_time:
            improved = False
            for i in range(n_local):
                if time.time() >= max_time:
                    break
                u = perm[i]
                # Compute V_row, V_col
                # V_row[k] = (b[v, :] - b[u, :])[p[k]] -- Wait, v is p[j], which varies.
                # Ah, V_row and V_col depend on v (which is p[j]).
                # So I cannot precompute V_row/V_col for fixed i independent of j.
                # My derivation above assumed V_row/V_col were fixed for the inner loop over j.
                # But V_row[k] = b[p[j], p[k]] - b[u, p[k]]. This depends on p[j].
                # So this vectorization trick doesn't work directly for varying j because V_row changes with j.
                
                # Back to the drawing board.
                # Is there another way?
                
                # Let's stick to the O(n) delta calculation but optimize it as much as possible.
                # Or use a different neighborhood (e.g., Or-opt: move a block).
                
                # Let's try to speed up the O(n) delta loop by using NumPy operations where possible,
                # but the loop over k is the bottleneck.
                
                # Alternative: Use a heuristic to limit the search.
                # Only try swaps that are likely to improve?
                
                # Let's try a simpler 2-opt with early termination and see if it's fast enough.
                # n=256. n^2 = 65k pairs. O(n) delta = 256 ops. Total 16M ops.
                # In Python, 16M simple ops might take ~1-2 seconds?
                # Let's test the speed of the inner loop.
                
                pi = perm[i]
                b_pi = b[pi]
                row_i = a[i]
                col_i = a[:, i]
                
                for j in range(i + 1, n_local):
                    if time.time() >= max_time:
                        break
                    
                    pj = perm[j]
                    b_pj = b[pj]
                    row_j = a[j]
                    col_j = a[j] # Wait, col_j is a[:, j]
                    col_j = a[:, j]
                    
                    # Compute delta
                    # We can vectorize the sum over k!
                    # diff_b_row = b_pj[perm] - b_pi[perm]  (shape n)
                    # diff_b_col = b[:, pj][perm] - b[:, pi][perm] (shape n)
                    
                    # sum_i = dot(row_i, diff_b_row) + dot(col_i, diff_b_col)
                    # sum_j = dot(row_j, -diff_b_row) + dot(col_j, -diff_b_col)
                    
                    # But we need to exclude k=i and k=j.
                    # Let's compute the full sum and subtract the terms for k=i and k=j.
                    
                    diff_b_row = b_pj[perm] - b_pi[perm]
                    diff_b_col = b[:, pj][perm] - b[:, pi][perm]
                    
                    # Full sum for i part:
                    sum_i_full = np.dot(row_i, diff_b_row) + np.dot(col_i, diff_b_col)
                    # Full sum for j part:
                    sum_j_full = np.dot(row_j, -diff_b_row) + np.dot(col_j, -diff_b_col)
                    
                    # Subtract k=i terms:
                    # For sum_i_full, term k=i:
                    # a[i,i] * diff_b_row[i] + a[i,i] * diff_b_col[i]
                    # diff_b_row[i] = b[pj, pi] - b[pi, pi]
                    # diff_b_col[i] = b[pi, pj] - b[pi, pi]
                    term_ii = a[i, i] * (b[pj, pi] - b[pi, pi]) + a[i, i] * (b[pi, pj] - b[pi, pi])
                    
                    # For sum_j_full, term k=i:
                    # -a[j,i] * diff_b_row[i] - a[i,j] * diff_b_col[i]
                    term_ji = -a[j, i] * (b[pj, pi] - b[pi, pi]) - a[i, j] * (b[pi, pj] - b[pi, pi])
                    
                    # Subtract k=j terms:
                    # For sum_i_full, term k=j:
                    # a[i,j] * diff_b_row[j] + a[j,i] * diff_b_col[j]
                    # diff_b_row[j] = b[pj, pj] - b[pi, pj]
                    # diff_b_col[j] = b[pj, pj] - b[pi, pj]
                    term_ij = a[i, j] * (b[pj, pj] - b[pi, pj]) + a[j, i] * (b[pj, pj] - b[pi, pj])
                    
                    # For sum_j_full, term k=j:
                    # -a[j,j] * diff_b_row[j] - a[j,j] * diff_b_col[j]
                    term_jj = -a[j, j] * (b[pj, pj] - b[pi, pj]) - a[j, j] * (b[pj, pj] - b[pi, pj])
                    
                    # The cross term (i,j) and (j,i) is handled separately in the original formula.
                    # Original formula:
                    # delta = sum_{k != i,j} [...] + a[i,j]*(b[pi,pj]-b[pj,pi]) + a[j,i]*(b[pj,pi]-b[pi,pj])
                    
                    # My vectorized sum includes k=i and k=j in the sums sum_i_full and sum_j_full.
                    # So:
                    # sum_{k != i,j} [...] = (sum_i_full - term_ii - term_ij) + (sum_j_full - term_ji - term_jj)
                    
                    # And the cross term:
                    cross = a[i, j] * (b[pi, pj] - b[pj, pi]) + a[j, i] * (b[pj, pi] - b[pi, pj])
                    cross = (a[i, j] - a[j, i]) * (b[pi, pj] - b[pj, pi])
                    
                    delta = (sum_i_full - term_ii - term_ij) + (sum_j_full - term_ji - term_jj) + cross
                    
                    if delta < 0:
                        perm[i], perm[j] = perm[j], perm[i]
                        current_cost += delta
                        improved = True
                        # Update pi, pj, etc? No, we are in the inner loop.
                        # But perm has changed. The precomputed diff_b_row etc are now invalid for subsequent j?
                        # No, diff_b_row depends on perm. If we swap, perm changes.
                        # So we must recompute diff_b_row for the next j?
                        # Actually, the loop is over j. If we swap i and j, then for the next j',
                        # the value at perm[i] is now the old perm[j].
                        # So the vectorized calculation for the next j' would be wrong if we don't update.
                        # But we are recalculating diff_b_row inside the j loop anyway.
                        # Wait, I put the calculation inside the j loop. So it's fine.
                        
                        # However, if we swap, the current_cost changes.
                        # And we should probably restart the i loop or continue?
                        # Standard 2-opt continues.
                        
                        # One issue: if we swap, the variables pi, pj, b_pi, b_pj, row_i, etc.
                        # are still from the old state for this specific i, j pair.
                        # But for the next j, we will recalculate everything based on the new perm.
                        # So it should be okay.
                        
                        # Actually, there is a subtle bug.
                        # If we swap perm[i] and perm[j], then for the next iteration of j (j+1),
                        # perm[i] has changed. But we are using `pi = perm[i]` which was set before the inner loop.
                        # So `pi` is stale.
                        # We need to update `pi` and `b_pi` etc. after a swap?
                        # Or better, re-fetch them inside the j loop?
                        # Let's move the fetching inside the j loop to be safe.
                        
                        # Let's restructure:
                        # for i:
                        #   for j:
                        #     pi = perm[i]
                        #     pj = perm[j]
                        #     ... compute delta ...
                        #     if delta < 0:
                        #        swap
                        #        current_cost += delta
                        #        improved = True
                        #        # break? or continue?
                        #        # If we break, we restart the i loop. This is safer and often effective.
                        #        break
                        #   if improved:
                        #      continue # restart i loop
                        #   else:
                        #      continue # next i
                        
                        # Let's try breaking out of the j loop on first improvement.
                        # This makes it a "first-improvement" 2-opt.
                        # It might be faster per pass but require more passes.
                        # Given the time limit, this might be better.
                        
                        # Let's implement first-improvement 2-opt.
                        pass

        return current_cost, perm

    # Re-implement two_opt_search with first-improvement and corrected logic
    def two_opt_search_v2(perm, current_cost, max_time):
        n_local = len(perm)
        while time.time() < max_time:
            improved = False
            for i in range(n_local):
                if time.time() >= max_time:
                    break
                for j in range(i + 1, n_local):
                    if time.time() >= max_time:
                        break
                    
                    pi = perm[i]
                    pj = perm[j]
                    
                    # Vectorized delta calculation
                    diff_b_row = b[pj, perm] - b[pi, perm]
                    diff_b_col = b[perm, pj] - b[perm, pi]
                    
                    sum_i_full = np.dot(a[i, :], diff_b_row) + np.dot(a[:, i], diff_b_col)
                    sum_j_full = np.dot(a[j, :], -diff_b_row) + np.dot(a[:, j], -diff_b_col)
                    
                    # Subtract k=i and k=j terms
                    # term_ii: a[i,i]*(b[pj,pi]-b[pi,pi]) + a[i,i]*(b[pi,pj]-b[pi,pi])
                    # term_ij: a[i,j]*(b[pj,pj]-b[pi,pj]) + a[j,i]*(b[pj,pj]-b[pi,pj])
                    # term_ji: -a[j,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pi,pj]-b[pi,pi])
                    # term_jj: -a[j,j]*(b[pj,pj]-b[pi,pj]) - a[j,j]*(b[pj,pj]-b[pi,pj])
                    
                    # Let's simplify the subtraction.
                    # The full sum includes k=i and k=j.
                    # We want sum_{k != i, j}.
                    
                    # Let's just compute the delta using the explicit formula for k=i, j and the vectorized sum for the rest?
                    # No, the vectorized sum is for ALL k.
                    
                    # Let's stick to the O(n) loop for the delta if vectorization is tricky.
                    # n=256 is small.
                    delta = 0
                    for k in range(n_local):
                        if k == i or k == j:
                            continue
                        pk = perm[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk])
                        delta += a[k, i] * (b[pk, pj] - b[pk, pi])
                        delta += a[j, k] * (b[pi, pk] - b[pj, pk])
                        delta += a[k, j] * (b[pk, pi] - b[pk, pj])
                    
                    delta += a[i, j] * (b[pi, pj] - b[pj, pi])
                    delta += a[j, i] * (b[pj, pi] - b[pi, pj])
                    
                    if delta < 0:
                        perm[i], perm[j] = perm[j], perm[i]
                        current_cost += delta
                        improved = True
                        break # First improvement, restart i loop
                if improved:
                    break # Restart i loop
            
            if not improved:
                break # Local minimum reached
        
        return current_cost, perm

    # Run 2-opt with time limit
    max_time = deadline - 5.0
    cost, p = two_opt_search_v2(p, best_cost, max_time)
    
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
