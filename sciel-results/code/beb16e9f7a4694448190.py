import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)
    
    # Precompute for delta calculation
    # For swap of positions i,j: delta = sum_k [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi]) 
    #                              + a[j,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pi]-b[pk,pj])
    #                              + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])]
    # Simplify: let pi=p[i], pj=p[j]
    # delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:]) + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
    #         + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    # But careful with double counting of i,j terms. Let's derive properly.
    
    # Actually, let me use a cleaner formulation:
    # Cost = sum_{i,j} a[i,j] * b[p[i],p[j]]
    # After swapping p[i] and p[j], the change is:
    # For pairs (i,k) and (k,i) where k != i,j:
    #   a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    # For pairs (j,k) and (k,j) where k != i,j:
    #   a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    # For pairs (i,j) and (j,i):
    #   a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    # For pairs (i,i) and (j,j): unchanged
    # For pair (i,j) already handled, (j,i) already handled
    # Note: the (i,k) and (j,k) terms for k != i,j can be combined:
    #   (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
    
    # Vectorized:
    # delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:]) + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
    #         + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    # But this double counts the i,j and j,i terms in the first two dot products.
    # The first dot product includes k=i and k=j:
    #   k=i: a[i,i]-a[j,i] times b[pj,pi]-b[pi,pi]
    #   k=j: a[i,j]-a[j,j] times b[pj,pj]-b[pi,pj]
    # The second dot product includes k=i and k=j:
    #   k=i: a[i,i]-a[i,j] times b[pi,pj]-b[pi,pi]
    #   k=j: a[j,i]-a[j,j] times b[pj,pj]-b[pj,pi]
    # These are messy. Let me just use the full O(n) computation which is still fast.
    
    def delta_2opt(p, i, j):
        """Compute cost delta of swapping positions i and j in permutation p. O(n)."""
        pi, pj = p[i], p[j]
        # Vectorized over all k
        # delta = sum_k [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi])
        #                        + a[j,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pi]-b[pk,pj])]
        #         + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        # Combine:
        # = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:]) + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
        #   - correction for double-counted i,j terms + explicit i,j pair terms
        # Let's just compute directly to avoid errors:
        pk = p  # all positions
        # Terms for k != i, j:
        # (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:]) gives sum over ALL k including i,j
        # We need to subtract the i and j contributions and add back correctly.
        # Actually, let me just do it the straightforward way - it's O(n) and fast enough.
        
        # Full O(n) computation:
        # For each k, the change in cost involving position i or j
        # This is: sum_k a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + sum_k a[k,i]*(b[p[k],pj]-b[p[k],pi])
        #         + sum_k a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + sum_k a[k,j]*(b[p[k],pi]-b[p[k],pj])
        # The k=i and k=j terms in these sums need careful handling.
        
        # Let me use the formula:
        # delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:]) + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
        # This double counts the (i,j) and (j,i) cross terms. Let's verify:
        # Term 1: sum_k (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
        #   k=i: (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
        #   k=j: (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
        # Term 2: sum_k (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        #   k=i: (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
        #   k=j: (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
        # 
        # The actual delta should be:
        # For k not in {i,j}:
        #   a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
        #   + a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])
        # = (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        # For k=i:
        #   a[i,i]*(b[pj,pi]-b[pi,pi]) + a[i,i]*(b[pi,pj]-b[pi,pi])  -- wait, this is wrong
        #   Actually for k=i, the terms are:
        #   a[i,i] changes from b[pi,pi] to b[pj,pi] for the a[i,i]*b[p[i],p[i]] term? No.
        #   p[i] changes from pi to pj. So a[i,i]*b[p[i],p[i]] changes from a[i,i]*b[pi,pi] to a[i,i]*b[pj,pj].
        #   Similarly a[i,k] for k=i: a[i,i]*b[p[i],p[i]] -> a[i,i]*b[pj,pj]
        #   a[k,i] for k=i: a[i,i]*b[p[i],p[i]] -> a[i,i]*b[pj,pj]
        #   So for k=i: delta_i = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,i]*(b[pj,pj]-b[pi,pi]) = 2*a[i,i]*(b[pj,pj]-b[pi,pi])?
        #   No wait, I'm confusing myself. Let me restart with a clear derivation.
        
        # Clear derivation:
        # Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
        # Swap p[i] and p[j]. Let pi=p[i], pj=p[j].
        # New: p'[i]=pj, p'[j]=pi, p'[k]=p[k] for k!=i,j.
        # Delta = sum_{u,v} a[u,v] * (b[p'[u],p'[v]] - b[p[u],p[v]])
        # Only terms where u or v is i or j change.
        # 
        # Case 1: u=i, v=j: a[i,j]*(b[pj,pi]-b[pi,pj])
        # Case 2: u=j, v=i: a[j,i]*(b[pi,pj]-b[pj,pi])
        # Case 3: u=i, v=k (k!=i,j): a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
        # Case 4: u=k, v=i (k!=i,j): a[k,i]*(b[p[k],pj]-b[p[k],pi])
        # Case 5: u=j, v=k (k!=i,j): a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
        # Case 6: u=k, v=j (k!=i,j): a[k,j]*(b[p[k],pi]-b[p[k],pj])
        # Case 7: u=i, v=i: a[i,i]*(b[pj,pj]-b[pi,pi])
        # Case 8: u=j, v=j: a[j,j]*(b[pi,pi]-b[pj,pj])
        #
        # Combine:
        # Cases 1+2: (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
        # Cases 3+5: sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
        # Cases 4+6: sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        # Cases 7+8: (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        #
        # Now, sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
        # = sum_{ALL k} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
        #   - (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
        #   - (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
        #
        # Similarly for the other sum.
        #
        # This is getting complex. Let me just use the direct O(n) computation which is clear and fast:
        
        # Direct O(n): iterate over all k
        # delta = 0
        # for k in range(n):
        #     if k == i or k == j: continue
        #     delta += a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + a[k,i]*(b[p[k],pj]-b[p[k],pi])
        #     delta += a[j,k]*(b[pi,p[k]]-b[pj,p[k]]) + a[k,j]*(b[p[k],pi]-b[p[k],pj])
        # delta += a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        # delta += a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
        
        # Vectorized version:
        # Let d1 = b[pj,:]-b[pi,:]  (row pj minus row pi of B)
        # Let d2 = b[:,pj]-b[:,pi]  (col pj minus col pi of B)
        # 
        # sum_k a[i,k]*d1[k] + a[k,i]*d2[k] + a[j,k]*(-d1[k]) + a[k,j]*(-d2[k])
        # = (a[i,:]-a[j,:]) @ d1 + (a[:,i]-a[:,j]) @ d2
        # But this includes k=i and k=j terms that need correction.
        
        d1 = b[pj] - b[pi]  # shape (n,)
        d2 = b[:, pj] - b[:, pi]  # shape (n,)
        
        # Sum over all k of (a[i,k]-a[j,k])*d1[k] + (a[k,i]-a[k,j])*d2[k]
        s1 = (a[i] - a[j]) @ d1
        s2 = (a[:, i] - a[:, j]) @ d2
        
        # Subtract k=i and k=j contributions from s1 and s2, then add correct terms
        # s1 includes k=i: (a[i,i]-a[j,i])*d1[i] = (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
        # s1 includes k=j: (a[i,j]-a[j,j])*d1[j] = (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
        # s2 includes k=i: (a[i,i]-a[i,j])*d2[i] = (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
        # s2 includes k=j: (a[j,i]-a[j,j])*d2[j] = (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
        
        # The correct delta for k=i:
        # From cases: u=i,v=i: a[i,i]*(b[pj,pj]-b[pi,pi])
        # u=i,v=j: a[i,j]*(b[pj,pi]-b[pi,pj])  -- but this is case 1, handled separately
        # u=j,v=i: a[j,i]*(b[pi,pj]-b[pj,pi])  -- case 2, handled separately
        # u=j,v=j: a[j,j]*(b[pi,pi]-b[pj,pj])
        # Wait, I think the cleanest is to just compute it directly without trying to be clever:
        
        # Let me use a different approach: compute the full cost change by
        # extracting the relevant rows/columns.
        
        # Actually, for n=256, even a simple loop is fast. But let me try the vectorized way:
        
        # delta = s1 + s2 - correction + explicit_i_j_terms
        # where correction removes the k=i,j terms from s1+s2 and we add back the correct ones.
        
        # k=i terms in s1+s2:
        # s1_k=i: (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
        # s2_k=i: (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
        # Correct k=i contribution to delta:
        # u=i,v=i: a[i,i]*(b[pj,pj]-b[pi,pi])
        # u=i,v=j: a[i,j]*(b[pj,pi]-b[pi,pj])  [case 1]
        # u=j,v=i: a[j,i]*(b[pi,pj]-b[pj,pi])  [case 2]
        # u=j,v=j: a[j,j]*(b[pi,pi]-b[pj,pj])
        # So for k=i in the "sum over k" part, we should NOT include u=i,v=j or u=j,v=i.
        
        # I think the simplest correct approach is:
        # delta = (a[i,:]-a[j,:]) @ d1 + (a[:,i]-a[:,j]) @ d2
        #         + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
        #         + a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
        #         - (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])  # remove from s1
        #         - (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])  # remove from s1
        #         - (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])  # remove from s2
        #         - (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])  # remove from s2
        
        # Let me verify with a small example... this is getting too complex. Let me just use the O(n) loop version which is clear and still fast for n=256.
        
        # For n=256, the O(n) delta computation with numpy vectorization:
        # We can vectorize over k by excluding i and j:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        pk = p[mask]
        ai = a[i, mask]
        aj = a[j, mask]
        a_i = a[mask, i]
        a_j = a[mask, j]
        
        # d1[k] = b[pj, p[k]] - b[pi, p[k]] for k in mask
        d1_masked = b[pj, pk] - b[pi, pk]
        d2_masked = b[pk, pj] - b[pk, pi]
        
        delta = ai @ d1_masked + a_i @ d2_masked + aj @ (-d1_masked) + a_j @ (-d2_masked)
        delta += a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
        delta += a[i, i] * (b[pj, pj] - b[pi, pi]) + a[j, j] * (b[pi, pi] - b[pj, pj])
        
        return delta
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
    
    # Initial assignment: Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 1: Hungarian-based local search
    hungarian_deadline = start_time + 35.0
    
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None
    
    hungarian_iter = 0
    while time.time() < hungarian_deadline:
        B_p = b[p]
        
        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        new_cost = calc_cost(p_new)
        
        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            hungarian_iter = 0
        else:
            # Perturb with random swaps
            improved = False
            for _ in range(30):
                i, j = np.random.choice(n, 2, replace=False)
                p_swap = p.copy()
                p_swap[i], p_swap[j] = p_swap[j], p_swap[i]
                d = delta_2opt(p, i, j)
                if d < 0:
                    p = p_swap
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if not improved:
                # More aggressive perturbation: random restart of a few elements
                if hungarian_iter > 5:
                    break
                # Randomly permute a subset
                subset = np.random.choice(n, min(10, n), replace=False)
                p_perm = p.copy()
                p_perm[subset] = np.random.permutation(p_perm[subset])
                new_cost = calc_cost(p_perm)
                if new_cost < current_cost:
                    p = p_perm
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                hungarian_iter += 1
    
    # Phase 2: Simulated Annealing with fast delta computation
    p = best_p.copy()
    current_cost = best_cost
    
    # Adaptive SA: estimate steps possible and set cooling accordingly
    remaining_time = deadline - time.time()
    # Estimate: each SA step takes ~5-10us with delta_2opt
    est_steps = int(remaining_time * 100000)  # ~100k steps per second
    T = 1000.0
    T_end = 0.1
    if est_steps > 100:
        alpha = (T_end / T) ** (1.0 / est_steps)
    else:
        alpha = 0.999
    
    sa_steps = 0
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        p[i], p[j] = p[j], p[i]
        d = delta_2opt(p, i, j)
        # Note: delta_2opt computes delta of swapping i,j in CURRENT p (after swap)
        # We need delta of swapping i,j in ORIGINAL p (before swap)
        # Let me fix this: compute delta BEFORE swapping
        
        # Actually, let me restructure: compute delta first, then decide
        p[i], p[j] = p[j], p[i]  # swap back
        d = delta_2opt(p, i, j)  # delta of swapping i,j in current p
        if d < 0 or np.random.rand() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        T *= alpha
        if T < T_end:
            T = T_end
        sa_steps += 1
        if sa_steps % 10000 == 0:
            # Periodic check
            pass
    
    # Phase 3: Final polish - steepest descent 2-opt
    polish_deadline = deadline
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for i in range(n):
            if time.time() > polish_deadline:
                break
            best_j = -1
            best_d = 0
            for j in range(i+1, n):
                d = delta_2opt(p, i, j)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 0:
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
    
    return best_p.tolist()
