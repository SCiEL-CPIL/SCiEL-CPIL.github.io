import numpy as np
import time
import sys

def solve(a, b):
    n = a.shape[0]
    # Time limit is 150s. We should aim to finish well before, e.g., 140s.
    start_time = time.time()
    deadline = start_time + 140.0
    
    # --- Phase 1: Karmarkar-Karp Constructive Heuristic ---
    # We maintain a matrix M where M[i, j] is the marginal cost of assigning 
    # facility i to location j, given the current partial assignment.
    # Initially, M is zeros? 
    # Actually, the initial marginal cost for assigning i to j is 0 
    # because there are no other assigned facilities.
    # But wait, the objective is sum_{u,v} A_{uv} B_{p(u), p(v)}.
    # If we assign i to j first, the cost added is A_{ii} B_{jj}.
    # So M[i, j] should start as A_{ii} B_{jj}? 
    # Let's trace:
    # Step 1: Assign i0 to j0. Cost += A[i0, i0] * B[j0, j0].
    # Step 2: Assign i1 to j1. Cost += A[i1, i1] B[j1, j1] + A[i1, i0] B[j1, j0] + A[i0, i1] B[j0, j1].
    # The update rule:
    # When we assign i to j, we add:
    # 1. A[i, i] * B[j, j] (self-term)
    # 2. Sum over all previously assigned pairs (k, p(k)): A[i, k] B[j, p(k)] + A[k, i] B[p(k), j]
    
    # We can maintain M[i, j] for unassigned i, j.
    # Initially M[i, j] = A[i, i] * B[j, j].
    
    M = a * b # This is not quite right. M[i,j] needs to be updated.
    # Let's initialize M properly.
    # M[i, j] represents the cost added if i is assigned to j NOW.
    # Initially, no one is assigned. So M[i, j] = A[i, i] * B[j, j].
    
    # However, A and B might not be symmetric. The problem statement implies 
    # standard QAP where A is flow and B is distance. Usually A is not symmetric 
    # but B is. The objective is sum A[i,j] B[p[i], p[j]].
    # If A is not symmetric, we must be careful.
    # The term is A[i,k] B[p[i], p[k]] + A[k,i] B[p[k], p[i]].
    
    # Let's use the standard KK update.
    # Unassigned facilities: F
    # Unassigned locations: L
    
    F = set(range(n))
    L = set(range(n))
    p = np.full(n, -1, dtype=np.int64)
    inv_p = np.full(n, -1, dtype=np.int64) # inv_p[location] = facility
    
    # Initialize M
    # M[i, j] = A[i, i] * B[j, j]
    # We can compute this vectorized.
    M = np.outer(np.diag(a), np.diag(b))
    
    # To speed up updates, we can maintain row/col sums or just use NumPy slicing.
    # n=150, so M is 150x150. Updating M is O(n^2) per step. Total O(n^3).
    
    for step in range(n):
        if time.time() > deadline:
            break
            
        # Find the minimum in M among unassigned i, j
        # Mask assigned rows and cols with inf
        # Since n is small, we can just loop or use advanced indexing.
        
        # Create a mask of available indices
        # For efficiency, we can just iterate. But let's try to vectorize the min selection.
        
        # Set M[i, j] = inf if i not in F or j not in L
        # We can maintain a boolean mask or just check.
        
        # Simple loop for selection is O(n^2) which is 22500 ops. 150 steps * 22500 = 3.3M ops. Fast.
        min_val = float('inf')
        best_i = -1
        best_j = -1
        
        # Optimization: Only look at unassigned
        # We can keep lists of unassigned i and j
        # But updating the list is overhead. 
        # Let's just scan the whole M matrix but ignore assigned.
        
        # Actually, we can maintain a list of unassigned facilities and locations.
        # Let's switch to lists for F and L to iterate faster.
        
        # Re-initializing F and L as lists for easier removal
        # We already have sets. Let's convert to lists for iteration?
        # Sets are fine for iteration.
        
        for i in F:
            # Only check unassigned locations
            for j in L:
                val = M[i, j]
                if val < min_val:
                    min_val = val
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            # Fallback if something went wrong (should not happen)
            break
            
        # Assign best_i to best_j
        p[best_i] = best_j
        inv_p[best_j] = best_i
        
        F.remove(best_i)
        L.remove(best_j)
        
        # Update M for remaining unassigned i', j'
        # M[i', j'] += A[i', best_i] * B[j', best_j] + A[best_i, i'] * B[best_j, j']
        
        # Get arrays of remaining indices
        rem_i = list(F)
        rem_j = list(L)
        
        if rem_i and rem_j:
            # Vectorized update
            # A_i_rem = A[rem_i, best_i] (shape len(rem_i))
            # B_j_rem = B[best_j, rem_j] (shape len(rem_j))
            # Term 1: A[i', best_i] * B[j', best_j]
            # This is outer product of A[rem_i, best_i] and B[best_j, rem_j]?
            # No.
            # We need to add to M[i', j'] the value:
            # A[i', best_i] * B[j', best_j] + A[best_i, i'] * B[best_j, j']
            
            # Let v1 = A[rem_i, best_i] (vector of size len(rem_i))
            # Let v2 = B[best_j, rem_j] (vector of size len(rem_j))
            # Term 1 matrix T1[i_idx, j_idx] = v1[i_idx] * v2[j_idx]
            
            # Let w1 = A[best_i, rem_i] (vector of size len(rem_i))
            # Let w2 = B[rem_j, best_j] (vector of size len(rem_j))
            # Term 2 matrix T2[i_idx, j_idx] = w1[i_idx] * w2[j_idx]
            
            # M[rem_i, rem_j] += T1 + T2
            
            # Note: M is indexed by original indices.
            # We need to update M at positions (i', j') for i' in rem_i, j' in rem_j.
            
            # Using advanced indexing:
            # M[np.ix_(rem_i, rem_j)] += np.outer(v1, v2) + np.outer(w1, w2)
            
            v1 = a[rem_i, best_i]
            v2 = b[best_j, rem_j]
            w1 = a[best_i, rem_i]
            w2 = b[rem_j, best_j]
            
            M[np.ix_(rem_i, rem_j)] += np.outer(v1, v2) + np.outer(w1, w2)
            
    # --- Phase 2: Local Search (2-opt) ---
    # If we broke out of KK early, p might have -1s. 
    # If so, we need to complete the assignment arbitrarily or restart.
    # Assuming KK completes, p is a valid permutation.
    
    if -1 in p:
        # Complete the permutation with remaining indices
        assigned_locs = set(p[p != -1])
        unassigned_locs = [j for j in range(n) if j not in assigned_locs]
        unassigned_fac = [i for i in range(n) if p[i] == -1]
        for i, j in zip(unassigned_fac, unassigned_locs):
            p[i] = j
        inv_p = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            inv_p[p[i]] = i

    # Calculate initial cost
    def calc_cost(p):
        # C = sum_{i,j} A[i,j] B[p[i], p[j]]
        # Using NumPy: np.einsum('ij,kl,ik,jl->', A, B, p, p) is not direct.
        # C = sum_i sum_j A[i,j] * B[p[i], p[j]]
        # Let P be the permutation matrix? No, p is the assignment.
        # We can do:
        # B_p = B[np.ix_(p, p)]
        # C = np.sum(A * B_p)
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    current_cost = calc_cost(p)
    
    # 2-opt Local Search
    # Swap i and k.
    # We will iterate until no improvement or time limit.
    
    # Pre-compute some terms to speed up delta calculation?
    # Delta for swapping i and k (locations j=p[i], l=p[k]):
    # Old: A[i,k]B[j,l] + A[k,i]B[l,j] + A[i,i]B[j,j] + A[k,k]B[l,l] + 
    #      sum_{v != i,k} (A[i,v]B[j,p[v]] + A[v,i]B[p[v],j] + A[k,v]B[l,p[v]] + A[v,k]B[p[v],l])
    # New: A[i,k]B[l,j] + A[k,i]B[j,l] + A[i,i]B[l,l] + A[k,k]B[j,j] +
    #      sum_{v != i,k} (A[i,v]B[l,p[v]] + A[v,i]B[p[v],l] + A[k,v]B[j,p[v]] + A[v,k]B[p[v],j])
    
    # Delta = New - Old
    # Terms involving only i,k:
    # D_self = (A[i,k]B[l,j] + A[k,i]B[j,l] + A[i,i]B[l,l] + A[k,k]B[j,j]) - 
    #          (A[i,k]B[j,l] + A[k,i]B[l,j] + A[i,i]B[j,j] + A[k,k]B[l,l])
    
    # Terms involving v:
    # D_v = sum_{v != i,k} ( A[i,v](B[l,p[v]] - B[j,p[v]]) + A[v,i](B[p[v],l] - B[p[v],j]) +
    #                        A[k,v](B[j,p[v]] - B[l,p[v]]) + A[v,k](B[p[v],j] - B[p[v],l]) )
    
    # We can precompute row/col sums of B weighted by A?
    # For a fixed i and candidate swap with k, we need to evaluate for all k.
    # This is O(n) per i, so O(n^2) per pass.
    # n=150, n^2=22500. Very fast.
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p.tolist()
            j = p[i]
            for k in range(i + 1, n):
                l = p[k]
                
                # Calculate Delta
                # D_self
                d_self = (a[i, k] * b[l, j] + a[k, i] * b[j, l] + a[i, i] * b[l, l] + a[k, k] * b[j, j]) - \
                         (a[i, k] * b[j, l] + a[k, i] * b[l, j] + a[i, i] * b[j, j] + a[k, k] * b[l, l])
                
                # D_v
                # We need sum over v != i, k
                # Let's compute the sum over all v and subtract v=i, v=k terms if necessary?
                # Or just compute directly.
                
                # Vectorized approach for D_v:
                # Let's create an array of p
                # We want sum_v (A[i,v] * (B[l, p[v]] - B[j, p[v]]) + ...)
                
                # This is getting complex to vectorize inside the loop efficiently without pre-computation.
                # Given n=150, a simple loop over v is 150 ops.
                # 150 (i) * 150 (k) * 150 (v) = 3.3M ops per pass.
                # If we do 100 passes, 330M ops. In Python, this might be too slow (seconds per pass).
                # We need to optimize.
                
                # Optimization: Precompute S1[i, l] = sum_v A[i,v] B[l, p[v]]
                # Then D_v part for i is S1[i, l] - S1[i, j].
                # Similarly for other terms.
                
                # Let's maintain a matrix S[i, loc] = sum_{v} A[i, v] B[loc, p[v]]
                # And T[loc, i] = sum_{v} B[loc, p[v]] A[v, i] ? No.
                
                # Let's define:
                # R[i, l] = sum_{v} A[i, v] B[l, p[v]]
                # C[l, i] = sum_{v} B[l, p[v]] A[v, i]  (Note: A[v,i] is column i)
                
                # Then:
                # Sum_v A[i,v] B[l, p[v]] = R[i, l]
                # Sum_v A[i,v] B[j, p[v]] = R[i, j]
                # Sum_v A[v,i] B[p[v], l] = ?
                #   Let u = p[v]. v = inv_p[u].
                #   Sum_u A[inv_p[u], i] B[u, l]
                #   This is like a matrix product.
                
                # This pre-computation updates are O(n^2) per swap.
                # Since we do O(n^2) swaps in a pass, total O(n^4) per pass? No.
                # If we update R and C after each swap, it's O(n^2) per swap.
                # Total O(n^4) per pass. 150^4 = 5e8. Too slow.
                
                # Alternative: Just compute D_v directly but vectorized for a fixed i, k?
                # For fixed i, k:
                # d_v = np.sum(
                #   a[i, :] * (b[l, p] - b[j, p]) + 
                #   a[:, i] * (b[p, l] - b[p, j]) + 
                #   a[k, :] * (b[j, p] - b[l, p]) + 
                #   a[:, k] * (b[p, j] - b[p, l])
                # )
                # Subtract terms where v=i or v=k to avoid double counting or self-terms?
                # The formula derived earlier excluded v=i, k.
                # If we sum over all v, we include v=i and v=k.
                # For v=i:
                # Term: A[i,i](B[l,p[i]] - B[j,p[i]]) + A[i,i](B[p[i],l] - B[p[i],j]) ...
                # p[i]=j.
                # A[i,i](B[l,j] - B[j,j]) + A[i,i](B[j,l] - B[j,j]) ...
                # This is messy.
                
                # Let's stick to the explicit sum over v != i, k but use NumPy slicing.
                # mask = np.ones(n, dtype=bool); mask[i]=False; mask[k]=False
                # v_idx = np.where(mask)[0]
                # This creates arrays. Overhead is high.
                
                # Given the time constraint, let's try a simpler 2-opt with direct calculation
                # and hope that n=150 is fast enough in NumPy if we don't loop v in Python.
                
                # We can compute the sum over all v and then subtract the v=i and v=k contributions.
                
                # Sum_all_v = np.sum(a[i, :] * (b[l, p] - b[j, p]) + a[:, i] * (b[p, l] - b[p, j]) + 
                #                    a[k, :] * (b[j, p] - b[l, p]) + a[:, k] * (b[p, j] - b[p, l]))
                
                # Correction for v=i:
                # Term_i = a[i, i] * (b[l, p[i]] - b[j, p[i]]) + a[i, i] * (b[p[i], l] - b[p[i], j]) + 
                #          a[k, i] * (b[j, p[i]] - b[l, p[i]]) + a[i, k] * (b[p[i], j] - b[p[i], l])
                # Note p[i] = j.
                # Term_i = a[i, i] * (b[l, j] - b[j, j]) + a[i, i] * (b[j, l] - b[j, j]) + 
                #          a[k, i] * (b[j, j] - b[l, j]) + a[i, k] * (b[j, j] - b[j, l])
                
                # Correction for v=k:
                # Term_k = a[i, k] * (b[l, p[k]] - b[j, p[k]]) + a[k, i] * (b[p[k], l] - b[p[k], j]) + 
                #          a[k, k] * (b[j, p[k]] - b[l, p[k]]) + a[k, k] * (b[p[k], j] - b[p[k], l])
                # Note p[k] = l.
                # Term_k = a[i, k] * (b[l, l] - b[j, l]) + a[k, i] * (b[l, l] - b[l, j]) + 
                #          a[k, k] * (b[j, l] - b[l, l]) + a[k, k] * (b[l, j] - b[l, l])
                
                # Delta_v = Sum_all_v - Term_i - Term_k
                
                # This involves 4 vector ops of size n (150). 
                # 4 * 150 = 600 ops per pair (i,k).
                # 11k pairs * 600 = 6.6M ops per pass.
                # Python overhead for NumPy calls might make this slow.
                # 11k NumPy calls per pass. 
                # If 1 pass takes 0.1s, we can do 100 passes in 10s.
                
                # Let's try it.
                
                # Calculate Sum_all_v
                # b_l_p = b[l, p]
                # b_j_p = b[j, p]
                # b_p_l = b[p, l]
                # b_p_j = b[p, j]
                
                # We can precompute these outside the k loop?
                # No, l changes with k.
                
                # Let's optimize by precomputing b_l_p etc for each k?
                # That's O(n^2) memory/time per i.
                
                # Let's just do it.
                
                b_l_p = b[l, p]
                b_j_p = b[j, p]
                b_p_l = b[p, l]
                b_p_j = b[p, j]
                
                sum_all = np.sum(a[i, :] * (b_l_p - b_j_p) + 
                                 a[:, i] * (b_p_l - b_p_j) + 
                                 a[k, :] * (b_j_p - b_l_p) + 
                                 a[:, k] * (b_p_j - b_p_l))
                
                # Term_i
                term_i = a[i, i] * (b[l, j] - b[j, j]) + a[i, i] * (b[j, l] - b[j, j]) + \
                         a[k, i] * (b[j, j] - b[l, j]) + a[i, k] * (b[j, j] - b[j, l])
                
                # Term_k
                term_k = a[i, k] * (b[l, l] - b[j, l]) + a[k, i] * (b[l, l] - b[l, j]) + \
                         a[k, k] * (b[j, l] - b[l, l]) + a[k, k] * (b[l, j] - b[l, l])
                
                d_v = sum_all - term_i - term_k
                
                delta = d_self + d_v
                
                if delta < 0:
                    # Swap
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    # Break inner loop to restart? Or continue?
                    # Continuing might use stale p for subsequent k.
                    # Better to break and restart the i loop or k loop.
                    # Let's break k loop and continue i loop.
                    break
            
    return p.tolist()

# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 140.0
    
    # --- Phase 1: Karmarkar-Karp Constructive Heuristic ---
    F = set(range(n))
    L = set(range(n))
    p = np.full(n, -1, dtype=np.int64)
    
    # Initialize M
    M = np.outer(np.diag(a), np.diag(b))
    
    for step in range(n):
        if time.time() > deadline:
            break
            
        min_val = float('inf')
        best_i = -1
        best_j = -1
        
        for i in F:
            for j in L:
                val = M[i, j]
                if val < min_val:
                    min_val = val
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break
            
        p[best_i] = best_j
        F.remove(best_i)
        L.remove(best_j)
        
        rem_i = list(F)
        rem_j = list(L)
        
        if rem_i and rem_j:
            v1 = a[rem_i, best_i]
            v2 = b[best_j, rem_j]
            w1 = a[best_i, rem_i]
            w2 = b[rem_j, best_j]
            
            M[np.ix_(rem_i, rem_j)] += np.outer(v1, v2) + np.outer(w1, w2)
            
    # Complete if necessary
    if -1 in p:
        assigned_locs = set(p[p != -1])
        unassigned_locs = [j for j in range(n) if j not in assigned_locs]
        unassigned_fac = [i for i in range(n) if p[i] == -1]
        for i, j in zip(unassigned_fac, unassigned_locs):
            p[i] = j

    # --- Phase 2: 2-opt Local Search ---
    def calc_cost(p):
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)

    current_cost = calc_cost(p)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p.tolist()
            j = p[i]
            for k in range(i + 1, n):
                l = p[k]
                
                # D_self
                d_self = (a[i, k] * b[l, j] + a[k, i] * b[j, l] + a[i, i] * b[l, l] + a[k, k] * b[j, j]) - \
                         (a[i, k] * b[j, l] + a[k, i] * b[l, j] + a[i, i] * b[j, j] + a[k, k] * b[l, l])
                
                # D_v
                b_l_p = b[l, p]
                b_j_p = b[j, p]
                b_p_l = b[p, l]
                b_p_j = b[p, j]
                
                sum_all = np.sum(a[i, :] * (b_l_p - b_j_p) + 
                                 a[:, i] * (b_p_l - b_p_j) + 
                                 a[k, :] * (b_j_p - b_l_p) + 
                                 a[:, k] * (b_p_j - b_p_l))
                
                term_i = a[i, i] * (b[l, j] - b[j, j]) + a[i, i] * (b[j, l] - b[j, j]) + \
                         a[k, i] * (b[j, j] - b[l, j]) + a[i, k] * (b[j, j] - b[j, l])
                
                term_k = a[i, k] * (b[l, l] - b[j, l]) + a[k, i] * (b[l, l] - b[l, j]) + \
                         a[k, k] * (b[j, l] - b[l, l]) + a[k, k] * (b[l, j] - b[l, l])
                
                d_v = sum_all - term_i - term_k
                
                delta = d_self + d_v
                
                if delta < 0:
                    p[i], p[k] = p[k], p[i]
                    current_cost += delta
                    improved = True
                    break
            
    return p.tolist()
# EVOLVE-BLOCK-END
