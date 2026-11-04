"""QAP solver for tho150 using spectral init + fast incremental local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # Safety margin before 150s kill

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    b_diag = np.diag(b_f)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        total = 0.0
        for i in range(n):
            total += np.dot(a_f[i], b_f[p[i], p])
        return total

    # --- Initialization Strategies ---

    def init_rank_matching():
        """Standard rank matching: heavy flows to central locations."""
        a_row_sums = a_f.sum(axis=1)
        b_row_sums = b_f.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def init_spectral():
        """Spectral initialization using eigenvectors."""
        try:
            # Center the matrices
            a_centered = a_f - a_f.mean()
            # Compute eigenvectors of A (flow matrix)
            eigvals, eigvecs = np.linalg.eigh(a_centered)
            idx = np.argsort(-np.abs(eigvals))
            spec_coords = eigvecs[:, idx[0]]
            
            # Rank facilities by spectral coordinate
            fac_rank = np.argsort(spec_coords, kind="stable")
            
            # Rank locations by distance sum (central ones first)
            loc_rank = np.argsort(b_f.sum(axis=1), kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[fac_rank] = loc_rank
            return p
        except:
            return init_rank_matching()

    def init_hungarian():
        """Hungarian algorithm on linearized cost matrix."""
        # Linearization: C[i, l] = sum_j a[i, j] * b_row_sums[j] ? 
        # Actually, a common heuristic is to assign based on row sums of A and B.
        # Let's try a slightly better linearization:
        # C[i, l] = sum_j a[i, j] * b[l, j] is not quite right for assignment.
        # Standard assignment relaxation:
        # min sum_i,l X[i,l] * C[i,l]
        # where C[i,l] = sum_j a[i,j] * b_row_sums[j] is a weak heuristic.
        # Better: C[i,l] = sum_j a[i,j] * b[l,j] ? No, b is symmetric usually but not guaranteed.
        # Let's stick to rank matching or spectral as they are robust.
        # However, we can use Hungarian on a matrix derived from the interaction.
        # Let's just use rank matching for now as it's fast and effective.
        return init_rank_matching()

    # --- Local Search with O(n) incremental updates ---
    
    def local_search(p, time_limit):
        p = p.copy()
        cost = compute_cost(p)
        
        # Precompute:
        # dot_row[i, l] = sum_k a[i, k] * b[l, p[k]]
        # dot_col[i, l] = sum_k a[k, i] * b[p[k], l]
        
        def recompute_precomp(p):
            b_col_p = b_f[:, p]  # (n, n): [l, k] = b[l, p[k]]
            b_row_p = b_f[p, :]  # (n, n): [k, l] = b[p[k], l]
            dot_row = a_f @ b_col_p.T
            dot_col = a_f.T @ b_row_p
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        # Use First Improvement for speed, but randomize order occasionally?
        # First Improvement is usually faster per iteration than Best Improvement.
        # We will do a pass of First Improvement. If no improvement, we stop.
        # To escape local minima, we might need restarts or simulated annealing.
        # Given the time limit, we will do multiple passes if time permits,
        # or restart with a perturbed solution.
        
        passes = 0
        max_passes = 1000
        
        while improved and time.time() < time_limit:
            improved = False
            passes += 1
            
            # First Improvement Pass
            for i in range(n):
                if time.time() > time_limit:
                    return p, cost
                pi = p[i]
                
                # We need to check all j > i to avoid double checking
                # But First Improvement usually just checks the first best one?
                # No, standard 2-opt First Improvement:
                # Iterate j. If swap(i,j) improves, do it and break inner loop (restart i loop or continue?).
                # If we do it and continue, the precomputed vectors for i are stale.
                # So if we swap, we must recompute precomp or break and restart.
                # Recomputing precomp is O(n^2). Doing it after every swap is O(n^3) per pass.
                # For n=150, n^2 = 22500. n^3 = 3.3M. 
                # If we do 100 swaps per pass, that's 330M ops. Might be tight.
                
                # Alternative: Update dot_row and dot_col incrementally in O(n).
                # Let's implement incremental update.
                
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        return p, cost
                    
                    pj = p[j]
                    if pi == pj:
                        continue
                    
                    # Compute delta
                    buu = b_diag[pi]
                    bvv = b_diag[pj]
                    buv = b_f[pi, pj]
                    bvu = b_f[pj, pi]
                    
                    delta = 0.0
                    # Corner terms
                    delta += a_f[i, i] * (bvv - buu)
                    delta += a_f[i, j] * (bvu - buv)
                    delta += a_f[j, i] * (buv - bvu)
                    delta += a_f[j, j] * (buu - bvv)
                    
                    # Interaction terms using precomputed dots
                    # Term 5: sum_{k not in {i,j}} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
                    # dot_row[i, l] = sum_k a[i, k] * b[l, p[k]]
                    # So sum_k a[i,k] * b[pj, p[k]] = dot_row[i, pj]
                    # sum_k a[i,k] * b[pi, p[k]] = dot_row[i, pi]
                    # We need to subtract k=i and k=j terms.
                    # k=i: a[i,i] * (b[pj, pi] - b[pi, pi]) = a[i,i]*(bvu - buu)
                    # k=j: a[i,j] * (b[pj, pj] - b[pi, pj]) = a[i,j]*(bvv - buv)
                    delta += (dot_row[i, pj] - dot_row[i, pi]) - a_f[i,i]*(bvu - buu) - a_f[i,j]*(bvv - buv)
                    
                    # Term 6: sum_{k not in {i,j}} a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
                    # = dot_row[j, pi] - dot_row[j, pj]
                    # Subtract k=i: a[j,i] * (b[pi, pi] - b[pj, pi]) = a[j,i]*(buu - bvu)
                    # Subtract k=j: a[j,j] * (b[pi, pj] - b[pj, pj]) = a[j,j]*(buv - bvv)
                    delta += (dot_row[j, pi] - dot_row[j, pj]) - a_f[j,i]*(buu - bvu) - a_f[j,j]*(buv - bvv)
                    
                    # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], pj] - b[p[k], pi])
                    # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
                    # sum_k a[k,i] * b[p[k], pj] = dot_col[i, pj]
                    # sum_k a[k,i] * b[p[k], pi] = dot_col[i, pi]
                    # Subtract k=i: a[i,i] * (b[pi, pj] - b[pi, pi]) = a[i,i]*(buv - buu)
                    # Subtract k=j: a[j,i] * (b[pj, pj] - b[pj, pi]) = a[j,i]*(bvv - bvu)
                    delta += (dot_col[i, pj] - dot_col[i, pi]) - a_f[i,i]*(buv - buu) - a_f[j,i]*(bvv - bvu)
                    
                    # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], pi] - b[p[k], pj])
                    # = dot_col[j, pi] - dot_col[j, pj]
                    # Subtract k=i: a[i,j] * (b[pi, pi] - b[pi, pj]) = a[i,j]*(buu - buv)
                    # Subtract k=j: a[j,j] * (b[pj, pi] - b[pj, pj]) = a[j,j]*(bvu - bvv)
                    delta += (dot_col[j, pi] - dot_col[j, pj]) - a_f[i,j]*(buu - buv) - a_f[j,j]*(bvu - bvv)
                    
                    if delta < -1e-9:
                        # Accept swap
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        
                        # Incremental update of dot_row and dot_col
                        # p[i] was pi, now pj. p[j] was pj, now pi.
                        
                        # Update dot_row:
                        # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
                        # For k not in {i,j}:
                        #   change in p[m] for m=i and m=j.
                        #   m=i: p[i] changed pi -> pj. Term a[k,i]*b[l, p[i]] changes by a[k,i]*(b[l,pj] - b[l,pi])
                        #   m=j: p[j] changed pj -> pi. Term a[k,j]*b[l, p[j]] changes by a[k,j]*(b[l,pi] - b[l,pj])
                        #   Total change for k: (a[k,i] - a[k,j]) * (b[l,pj] - b[l,pi])
                        #   Let diff_b_l = b[:, pj] - b[:, pi] (vector of length n)
                        #   dot_row[k, :] += (a[k,i] - a[k,j]) * diff_b_l
                        #   This is O(n) for each k. Total O(n^2).
                        
                        # For k=i:
                        #   dot_row[i, l] = sum_m a[i,m] * b[l, p[m]]
                        #   m=i: a[i,i]*b[l, pj] (was pi) -> change a[i,i]*(b[l,pj] - b[l,pi])
                        #   m=j: a[i,j]*b[l, pi] (was pj) -> change a[i,j]*(b[l,pi] - b[l,pj])
                        #   Total: (a[i,i] - a[i,j]) * (b[l,pj] - b[l,pi])
                        
                        # For k=j:
                        #   dot_row[j, l] = sum_m a[j,m] * b[l, p[m]]
                        #   m=i: a[j,i]*b[l, pj] (was pi) -> change a[j,i]*(b[l,pj] - b[l,pi])
                        #   m=j: a[j,j]*b[l, pi] (was pj) -> change a[j,j]*(b[l,pi] - b[l,pj])
                        #   Total: (a[j,i] - a[j,j]) * (b[l,pj] - b[l,pi])
                        
                        diff_b = b_f[:, pj] - b_f[:, pi] # (n,)
                        
                        # Vectorized update for all k
                        # delta_a = a[:, i] - a[:, j]
                        # dot_row += np.outer(delta_a, diff_b)
                        
                        delta_a_row = a_f[:, i] - a_f[:, j]
                        dot_row += np.outer(delta_a_row, diff_b)
                        
                        # Update dot_col:
                        # dot_col[k, l] = sum_m a[m, k] * b[p[m], l]
                        # For k not in {i,j}:
                        #   m=i: a[i,k]*b[pj, l] (was pi) -> change a[i,k]*(b[pj,l] - b[pi,l])
                        #   m=j: a[j,k]*b[pi, l] (was pj) -> change a[j,k]*(b[pi,l] - b[pj,l])
                        #   Total: (a[i,k] - a[j,k]) * (b[pj,l] - b[pi,l])
                        #   Let diff_bT_l = b[pj, :] - b[pi, :]
                        #   dot_col[:, k] += (a[i,k] - a[j,k]) * diff_bT
                        #   This is O(n) for each k. Total O(n^2).
                        
                        # For k=i:
                        #   dot_col[i, l] = sum_m a[m, i] * b[p[m], l]
                        #   m=i: a[i,i]*b[pj, l] (was pi) -> change a[i,i]*(b[pj,l] - b[pi,l])
                        #   m=j: a[j,i]*b[pi, l] (was pj) -> change a[j,i]*(b[pi,l] - b[pj,l])
                        #   Total: (a[i,i] - a[j,i]) * (b[pj,l] - b[pi,l])
                        
                        # For k=j:
                        #   dot_col[j, l] = sum_m a[m, j] * b[p[m], l]
                        #   m=i: a[i,j]*b[pj, l] (was pi) -> change a[i,j]*(b[pj,l] - b[pi,l])
                        #   m=j: a[j,j]*b[pi, l] (was pj) -> change a[j,j]*(b[pi,l] - b[pj,l])
                        #   Total: (a[i,j] - a[j,j]) * (b[pj,l] - b[pi,l])
                        
                        diff_bT = b_f[pj, :] - b_f[pi, :] # (n,)
                        
                        # delta_a_col = a[i, :] - a[j, :]
                        # dot_col += np.outer(delta_a_col, diff_bT)
                        
                        delta_a_col = a_f[i, :] - a_f[j, :]
                        dot_col += np.outer(delta_a_col, diff_bT)
                        
                        # After swap, p[i] is pj, p[j] is pi.
                        # The inner loop `for j in range(i+1, n)` continues with stale pi/pj?
                        # No, we broke out of the logic? 
                        # If we do First Improvement, we should break the inner loop 
                        # and restart the outer loop? Or just continue?
                        # If we continue, the vectors for i are updated, but 
                        # the `pi` variable in the loop is stale? 
                        # We set pi = p[i] at the start of the i-loop.
                        # After swap, p[i] changed. So pi is stale.
                        # Also, the j-loop is iterating over j. 
                        # If we just continue, we will check j+1 with stale pi.
                        # So we must break the inner loop and restart the outer loop?
                        # Or re-fetch pi?
                        # If we re-fetch pi, we still have the issue that 
                        # the precomputed vectors for the *current* i are updated, 
                        # but the loop structure is fragile.
                        
                        # Safest: Break inner loop, and restart outer loop from 0?
                        # Or just break inner loop and continue outer loop (i+1)?
                        # If we continue outer loop, we miss potential improvements involving i again.
                        # Standard First Improvement often just continues. 
                        # Let's break inner loop and let the outer loop continue.
                        # The next i will use fresh precomputed vectors (which are globally updated).
                        # This is a valid local search strategy (though not strictly 
                        # "restart from 0", it's a single pass).
                        break
                        
            # If we finished a pass without improvement, we are at a local optimum.
            # We can try to perturb and restart to escape.
            if not improved:
                # Simple perturbation: swap 2 random elements
                if time.time() < time_limit - 10.0:
                    i1, i2 = np.random.randint(0, n, 2)
                    if i1 != i2:
                        p[i1], p[i2] = p[i2], p[i1]
                        # Recompute cost and precomp? 
                        # Cost is not updated. We should recompute cost to be safe.
                        cost = compute_cost(p)
                        dot_row, dot_col = recompute_precomp(p)
                        improved = True # Force another pass
                        # Limit perturbations?
                        # We'll just let the loop run.
        
        return p, cost

    # --- Main Logic ---
    
    # Try a few different initializations
    best_p = None
    best_cost = np.inf
    
    # 1. Rank Matching
    p0 = init_rank_matching()
    p_sol, c_sol = local_search(p0, deadline - 5.0)
    if c_sol < best_cost:
        best_cost = c_sol
        best_p = p_sol.copy()
        
    # 2. Spectral
    if time.time() < deadline - 20.0:
        p1 = init_spectral()
        p_sol, c_sol = local_search(p1, deadline - 5.0)
        if c_sol < best_cost:
            best_cost = c_sol
            best_p = p_sol.copy()

    # 3. Random Initialization (if time permits)
    if time.time() < deadline - 20.0:
        p2 = np.random.permutation(n)
        p_sol, c_sol = local_search(p2, deadline - 5.0)
        if c_sol < best_cost:
            best_cost = c_sol
            best_p = p_sol.copy()

    # Final Polish: Run local search on the best found solution again with remaining time
    if time.time() < deadline - 2.0:
        p_final, c_final = local_search(best_p, deadline - 0.5)
        if c_final < best_cost:
            best_p = p_final
            best_cost = c_final

    return best_p.tolist()

# EVOLVE-BLOCK-END
