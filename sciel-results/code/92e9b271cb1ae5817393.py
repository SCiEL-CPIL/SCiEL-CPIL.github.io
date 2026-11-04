import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    # Greedy rank-matching initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    B_p = b[np.ix_(p, p)].copy()
    
    def two_opt(p, cost, B_p, max_passes=200):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline:
                return p, cost, B_p
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost, B_p
                for j in range(i + 1, n):
                    pi = p[i]
                    pj = p[j]
                    
                    # Delta for swapping facilities i and j
                    # New cost - old cost when we swap p[i] and p[j]
                    # delta = sum_k (a[i,k] - a[j,k]) * (b[pj,k] - b[pi,k])
                    #       + sum_k (a[k,i] - a[k,j]) * (b[k,pj] - b[k,pi])
                    #       - (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
                    #       - (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])  # correction for double count
                    # Actually let's be more careful.
                    
                    # Let's use the standard formula:
                    # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj,k] - b[pi,k])
                    #       + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[k,pj] - b[k,pi])
                    #       + (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
                    #       + (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
                    #       - (a[i,i] - a[j,i]) * (b[pj,i] - b[pi,i])  # but p[i]=pi, p[j]=pj
                    #       ...
                    
                    # Simpler vectorized approach:
                    # term1 = (a[i,:] - a[j,:]) @ (B_p[pj,:] - B_p[pi,:])  -- but B_p is indexed by position, not facility
                    # B_p[i,k] = b[p[i], p[k]]
                    # So B_p[:, pj] = b[p[:], p[j]] = b[p[:], pj]
                    # B_p[:, pi] = b[p[:], p[i]] = b[p[:], pi]
                    
                    # delta = sum_k (a[i,k] - a[j,k]) * (b[pj,k] - b[pi,k])
                    #       + sum_k (a[k,i] - a[k,j]) * (b[k,pj] - b[k,pi])
                    #       - (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
                    #       - (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
                    
                    # Wait, b is symmetric in QAP? No, not necessarily.
                    # Let me just compute it directly with vectors.
                    
                    # Let u = a[i,:] - a[j,:]
                    # Let v = a[:,i] - a[:,j]
                    # Let w = b[:,pj] - b[:,pi]  (column vectors)
                    # Let x = b[pj,:] - b[pi,:]  (row vectors)
                    
                    # delta = u @ x + v @ w - (a[i,j] - a[j,i])*(b[pi,pj] - b[pj,pi]) - (a[i,j] - a[j,i])*(b[pj,pi] - b[pi,pj])
                    # Hmm, this is getting complicated. Let me just use a simpler but correct formula.
                    
                    # The change in cost when swapping positions i and j:
                    # Old contribution involving i or j:
                    #   sum_k a[i,k]*b[pi,pk] + sum_k a[k,i]*b[pk,pi] 
                    #   + sum_k a[j,k]*b[pj,pk] + sum_k a[k,j]*b[pk,pj]
                    #   (note: a[i,j]*b[pi,pj] and a[j,i]*b[pj,pi] are counted in both)
                    
                    # New contribution:
                    #   sum_k a[i,k]*b[pj,pk] + sum_k a[k,i]*b[pk,pj]
                    #   + sum_k a[j,k]*b[pi,pk] + sum_k a[k,j]*b[pk,pi]
                    
                    # delta = new - old
                    # = sum_k (a[i,k]-a[j,k])*b[pj,pk] - sum_k (a[i,k]-a[j,k])*b[pi,pk]
                    #   + sum_k (a[k,i]-a[k,j])*b[pk,pj] - sum_k (a[k,i]-a[k,j])*b[pk,pi]
                    # = (a[i,:]-a[j,:]) @ b[pj,:] - (a[i,:]-a[j,:]) @ b[pi,:]
                    #   + (a[:,i]-a[:,j]) @ b[:,pj] - (a[:,i]-a[:,j]) @ b[:,pi]
                    
                    diff_a_row = a[i,:] - a[j,:]
                    diff_a_col = a[:,i] - a[:,j]
                    
                    s1 = np.dot(diff_a_row, b[pj,:] - b[pi,:])
                    s2 = np.dot(diff_a_col, b[:,pj] - b[:,pi])
                    
                    # But we double-counted the (i,j) and (j,i) terms in both s1 and s2
                    # In s1: k=j term: a[i,j]*b[pj,pj] - a[j,j]*b[pj,pj] - a[i,j]*b[pi,pj] + a[j,j]*b[pi,pj]
                    #         = (a[i,j]-a[j,j])*b[pj,pj] - (a[i,j]-a[j,j])*b[pi,pj]
                    # In s2: k=j term: a[j,i]*b[pj,pj] - a[j,j]*b[pj,pj] - a[j,i]*b[pj,pi] + a[j,j]*b[pj,pi]
                    #         = (a[j,i]-a[j,j])*b[pj,pj] - (a[j,i]-a[j,j])*b[pj,pi]
                    
                    # Actually, let's just compute delta directly for correctness:
                    # delta = s1 + s2 - correction
                    # The correction accounts for the fact that when k=i or k=j, we're swapping
                    # the same pair, so we need to subtract the double-counted terms.
                    
                    # Let's just use the straightforward O(n) computation per pair:
                    # This is O(n^3) total per pass, which for n=256 is 256^3 = 16.7M ops per pass
                    # That's too slow. Let me stick with the vectorized approach but verify.
                    
                    # Actually, the standard correct formula is:
                    # delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:]) 
                    #       + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
                    #       - 2*(a[i,j]-a[j,i])*(b[pi,pj]-b[pj,pi])  -- NO this isn't right either
                    
                    # Let me just compute it the safe way with precomputed arrays:
                    # We have B_p where B_p[k,l] = b[p[k], p[l]]
                    # b[pj,:] - b[pi,:] is a row of b
                    # b[:,pj] - b[:,pi] is a column of b
                    
                    # The formula delta = s1 + s2 is actually correct IF we account for:
                    # s1 includes k=i and k=j terms
                    # s2 includes k=i and k=j terms
                    # When k=i in s1: (a[i,i]-a[j,i])*b[pj,pi] - (a[i,i]-a[j,i])*b[pi,pi]
                    # When k=j in s1: (a[i,j]-a[j,j])*b[pj,pj] - (a[i,j]-a[j,j])*b[pi,pj]
                    # When k=i in s2: (a[i,i]-a[i,j])*b[pi,pj] - (a[i,i]-a[i,j])*b[pi,pi]  
                    # When k=j in s2: (a[j,i]-a[j,j])*b[pj,pj] - (a[j,i]-a[j,j])*b[pj,pi]
                    
                    # The correct delta should be:
                    # sum_{k != i,j} [(a[i,k]-a[j,k])*b[pj,pk] + (a[k,i]-a[k,j])*b[pk,pj]]
                    # + (a[i,j]-a[j,i])*b[pj,pi] + (a[j,i]-a[i,j])*b[pi,pj]
                    # - sum_{k != i,j} [(a[i,k]-a[j,k])*b[pi,pk] + (a[k,i]-a[k,j])*b[pk,pi]]
                    # - (a[i,j]-a[j,i])*b[pi,pj] - (a[j,i]-a[i,j])*b[pj,pi]
                    
                    # = sum_{k != i,j} [(a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk]) + (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])]
                    # + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj]) + (a[j,i]-a[i,j])*(b[pi,pj]-b[pj,pi])
                    # = sum_{k != i,j} [(a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk]) + (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])]
                    # + 2*(a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    
                    # Now s1 = sum_k (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
                    #    = sum_{k!=i,j} ... + (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) + (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    # s2 = sum_k (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])
                    #    = sum_{k!=i,j} ... + (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi]) + (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    
                    # So s1+s2 = sum_{k!=i,j}[...] + (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) + (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    #          + (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi]) + (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    
                    # delta = (s1+s2) - (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) - (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    #        - (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi]) - (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    #        + 2*(a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    
                    # This is getting very complex. Let me just use a simpler O(n) per swap check
                    # but vectorize over j for each i.
                    
                    # Actually, let me just use the direct computation which is O(n) per (i,j) pair:
                    # delta = sum_k [(a[i,k]-a[j,k])*(b[pj,k]-b[pi,k]) + (a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])]
                    #         - (a[i,i]-a[j,i])*(b[pj,i]-b[pi,i]) - (a[i,j]-a[j,j])*(b[pj,j]-b[pi,j])
                    #         - (a[i,i]-a[i,j])*(b[i,pj]-b[i,pi]) - (a[j,i]-a[j,j])*(b[j,pj]-b[j,pi])
                    #         + 2*(a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    
                    # For now, let me just use a simpler but potentially less efficient approach:
                    # Compute the full cost change by only considering affected terms.
                    
                    # Let me just use the most straightforward correct method:
                    delta = 0
                    # Terms where one index is i or j
                    # Old: a[i,k]*b[pi,pk] for all k, a[k,i]*b[pk,pi] for all k
                    #      a[j,k]*b[pj,pk] for all k, a[k,j]*b[pk,pj] for all k
                    # New: a[i,k]*b[pj,pk] for all k, a[k,i]*b[pk,pj] for all k
                    #      a[j,k]*b[pi,pk] for all k, a[k,j]*b[pk,pi] for all k
                    
                    # delta = sum_k (a[i,k]-a[j,k])*(b[pj,k]-b[pi,k])
                    #       + sum_k (a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])
                    #       - (a[i,j]-a[j,i])*(b[pi,pj]-b[pj,pi])  # remove double count of (i,j) in first sum
                    #       - (a[j,i]-a[i,j])*(b[pj,pi]-b[pi,pj])  # remove double count of (j,i) in second sum
                    #       Hmm, I'm going in circles.
                    
                    # Let me just compute it directly:
                    # For each k, the contribution changes from:
                    #   a[i,k]*b[pi,pk] + a[k,i]*b[pk,pi] + a[j,k]*b[pj,pk] + a[k,j]*b[pk,pj]
                    # to:
                    #   a[i,k]*b[pj,pk] + a[k,i]*b[pk,pj] + a[j,k]*b[pi,pk] + a[k,j]*b[pk,pi]
                    # BUT when k=i or k=j, we're double counting.
                    
                    # The cleanest approach: just compute the two dot products and subtract corrections.
                    # I'll trust the formula: delta = s1 + s2 where
                    # s1 = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:])
                    # s2 = (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
                    # and then subtract the over-counted terms.
                    
                    # After much deliberation, the correct formula is:
                    # delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:])
                    #       + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])
                    #       - (a[i,j]-a[j,i])*(b[pi,pj]-b[pj,pi])
                    #       - (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    #       - (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])  -- no this isn't right
                    #       ...
                    
                    # I'll just use the O(n) direct computation for correctness, accepting the speed cost:
                    # This makes each pass O(n^3) = 256^3 ≈ 16.7M, which at ~1ns/op is ~17ms per pass
                    # With 200 passes that's 3.4s, which is acceptable.
                    
                    # Actually no, let me just benchmark the vectorized version.
                    # The dot products s1 and s2 are O(n) each, so O(n) per (i,j) pair.
                    # Total per pass: O(n^3) = same.
                    
                    # Let me just go with the vectorized dot product approach and add the correction:
                    s1 = np.dot(diff_a_row, b[pj,:] - b[pi,:])
                    s2 = np.dot(diff_a_col, b[:,pj] - b[:,pi])
                    
                    # Correction for the (i,j) and (j,i) terms being counted in both:
                    # In s1, k=j: (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    # In s2, k=j: (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    # In s1, k=i: (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
                    # In s2, k=i: (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
                    
                    # The true delta for the (i,j) interaction:
                    # Old: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
                    # New: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
                    # Change: a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
                    #       = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    
                    # The true delta for the (i,i) interaction:
                    # Old: a[i,i]*b[pi,pi]
                    # New: a[i,i]*b[pj,pj]  -- NO, facility i moves to pj, so b[pj,pj]
                    # Change: a[i,i]*(b[pj,pj]-b[pi,pi])
                    
                    # The true delta for the (j,j) interaction:
                    # Old: a[j,j]*b[pj,pj]
                    # New: a[j,j]*b[pi,pi]
                    # Change: a[j,j]*(b[pi,pi]-b[pj,pj])
                    
                    # So total delta = sum_{k!=i,j} [(a[i,k]-a[j,k])*(b[pj,k]-b[pi,k]) + (a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])]
                    #                + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    #                + a[i,i]*(b[pj,pj]-b[pi,pi])
                    #                + a[j,j]*(b[pi,pi]-b[pj,pj])
                    
                    # s1 = sum_{k!=i,j}(a[i,k]-a[j,k])*(b[pj,k]-b[pi,k])
                    #    + (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi])
                    #    + (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    # s2 = sum_{k!=i,j}(a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])
                    #    + (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
                    #    + (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    
                    # s1+s2 = sum_{k!=i,j}[...]
                    #       + (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) + (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    #       + (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi]) + (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    
                    # delta = s1+s2
                    #       - (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) - (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
                    #       - (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi]) - (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
                    #       + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    #       + a[i,i]*(b[pj,pj]-b[pi,pi])
                    #       + a[j,j]*(b[pi,pi]-b[pj,pj])
                    
                    # Let me simplify:
                    # coeff of (b[pj,pi]-b[pi,pi]): -(a[i,i]-a[j,i]) = -a[i,i]+a[j,i]
                    # coeff of (b[pj,pj]-b[pi,pj]): -(a[i,j]-a[j,j]) = -a[i,j]+a[j,j]
                    # coeff of (b[pi,pj]-b[pi,pi]): -(a[i,i]-a[i,j]) = -a[i,i]+a[i,j]
                    # coeff of (b[pj,pj]-b[pj,pi]): -(a[j,i]-a[j,j]) = -a[j,i]+a[j,j]
                    # Plus: (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    #      + a[i,i]*(b[pj,pj]-b[pi,pi])
                    #      + a[j,j]*(b[pi,pi]-b[pj,pj])
                    
                    # This is extremely complex. Let me just use the O(n) direct computation.
                    # For n=256, O(n^3) per pass is fine.
                    
                    # Direct O(n) computation:
                    # delta = 0
                    # for k in range(n):
                    #     if k == i or k == j:
                    #         continue
                    #     delta += (a[i,k]-a[j,k])*(b[pj,k]-b[pi,k]) + (a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])
                    # delta += (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                    # delta += a[i,i]*(b[pj,pj]-b[pi,pi])
                    # delta += a[j,j]*(b[pi,pi]-b[pj,pj])
                    
                    # But this is a Python loop which is slow. Let me vectorize it:
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    s1_masked = np.dot(diff_a_row[mask], b[pj,:][mask] - b[pi,:][mask])
                    s2_masked = np.dot(diff_a_col[mask], b[:,pj][mask] - b[:,pi][mask])
                    delta = s1_masked + s2_masked
                    delta += (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
                    delta += a[i,i] * (b[pj,pj] - b[pi,pi])
                    delta += a[j,j] * (b[pi,pi] - b[pj,pj])
                    
                    if delta < -1e-9:
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        B_p[i,:] = b[p[i], p]
                        B_p[j,:] = b[p[j], p]
                        B_p[:,i] = b[p, p[i]]
                        B_p[:,j] = b[p, p[j]]
                        improved = True
                        break
            # End for i
        return p, cost, B_p
    
    # Main ILS loop
    while time.time() < deadline:
        p, current_cost, B_p = two_opt(p, current_cost, B_p, max_passes=100)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() > deadline - 5.0:
            break
        
        # Hungarian re-optimization
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        C = a @ b_col_p + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        new_cost = compute_cost(p_new)
        
        if new_cost < current_cost:
            p = p_new
            current_cost = new_cost
            B_p = b[np.ix_(p, p)].copy()
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Perturbation: random swaps
            num_swaps = random.randint(5, 15)
            for _ in range(num_swaps):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            
            current_cost = compute_cost(p)
            B_p = b[np.ix_(p, p)].copy()
    
    return best_p.tolist()
