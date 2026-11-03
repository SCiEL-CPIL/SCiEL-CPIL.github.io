"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Precompute cost function components
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i a[i,i]*b[p[i],p[i]] + sum_{i != j} a[i,j]*b[p[i],p[j]]
    
    def compute_cost(p):
        # p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized: for each i, sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] @ b[p[i], :]) ... wait, that's sum_i sum_j a[i,j] b[p[i],p[j]]
        # = sum_i a[i,:] dot b[p[i],:]
        # But b[p[i],:] is row p[i] of b
        # So cost = sum_i np.dot(a[i,:], b[p[i],:])
        # This is O(n^2) per evaluation with vectorization
        
        # More efficient: cost = trace(A * B_p) where B_p[i,j] = b[p[i],p[j]]
        # Actually: sum_{i,j} a[i,j] * b[p[i],p[j]]
        # Let's use: for each i, the contribution is a[i,:] @ b[p[i],:]
        # Sum over i
        
        # Efficient computation using matrix operations:
        # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] * b[p[i], :]).sum()
        
        # Even better: think of it as
        # Let P be the permutation matrix where P[i, p[i]] = 1
        # Then b[p[i], p[j]] = (P^T B P)[i, j]
        # cost = sum_{i,j} a[i,j] * (P^T B P)[i,j] = trace(A * P^T B P) = trace(P A P^T B)
        
        # For computation with permutation vector p:
        # cost = np.sum(a * b[np.ix_(p, p)])
        # But that creates a full n x n array which is expensive for n=150
        
        # Alternative: cost = sum_i (a[i,:] @ b[p[i], :])
        # = sum_i dot(a[i,:], b[p[i],:])
        
        # Most vectorized: 
        # cost = np.einsum('ij,ij->', a, b[np.ix_(p, p)])
        # Still creates n x n temporary
        
        # Better: 
        # cost = 0
        # for i in range(n):
        #     cost += np.dot(a[i,:], b[p[i],:])
        # This is n dot products of length n, so O(n^2) total
        
        # Or: cost = np.sum(a * b[np.ix_(p,p)])  - creates n^2 array
        
        # Let's use the loop version but with numpy for each row:
        return int(np.sum([np.dot(a[i,:], b[p[i],:]) for i in range(n)]))
    
    def compute_cost_fast(p):
        """Faster cost computation using matrix multiply trick."""
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # = trace(A @ B_p) where B_p[i,j] = b[p[i], p[j]]
        # B_p = B[p][:, p] ... no, B[p][:, p] means rows indexed by p, cols indexed by p
        # Actually B[p][:, p] is not quite right.
        # b[p[i], p[j]] for all i,j = b[np.ix_(p, p)]
        # cost = np.sum(a * b[np.ix_(p, p)])
        
        # Alternative using trace:
        # Let's define C where C[i,j] = b[p[i], p[j]]
        # cost = sum_{i,j} a[i,j] C[i,j]
        
        # Using the fact that if we form the permutation matrix:
        # cost = trace(A @ B @ P @ P^T) ... this is getting complex
        
        # Simplest fast version:
        # Pre-compute nothing, just:
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))
    
    # Initial heuristic: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost_fast(p)
    
    # Local search with 2-opt style swaps
    # For a swap of locations of facilities i and j:
    # delta = cost_after - cost_before
    # 
    # Let's derive the delta for swapping p[i] and p[j]:
    # Before: terms involving i or j in the sum
    # After: p'[i] = p[j], p'[j] = p[i], others unchanged
    # 
    # The change in cost:
    # Only terms where at least one index is i or j change
    # 
    # Let's compute delta = cost_after - cost_before
    # 
    # cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # 
    # Terms that change: those with u in {i,j} or v in {i,j}
    # 
    # This is complex. Let's use a different approach.
    # 
    # For each pair (i,j), compute the delta directly by checking which terms change.
    # 
    # Actually, let's use a simpler but effective local search:
    # Try all 2-swaps, compute delta efficiently.
    
    # Precompute row and column sums of b for faster delta computation
    # 
    # When we swap p[i] and p[j]:
    # Let li = p[i], lj = p[j]
    # 
    # The terms that change are:
    # 1. a[i,i]*b[li,li] -> a[i,i]*b[lj,lj]
    # 2. a[j,j]*b[lj,lj] -> a[j,j]*b[li,li]
    # 3. For k not in {i,j}:
    #    a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]
    #    a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]
    #    a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]
    #    a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]
    # 4. a[i,j]*b[li,lj] -> a[i,j]*b[lj,li]
    # 5. a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
    
    # delta = sum of (new - old) for all these terms
    
    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        delta = 0
        
        # Diagonal terms
        delta += a[i,i] * (b[lj,lj] - b[li,li])
        delta += a[j,j] * (b[li,li] - b[lj,lj])
        
        # Cross terms between i and j
        delta += a[i,j] * (b[lj,li] - b[li,lj])
        delta += a[j,i] * (b[li,lj] - b[lj,li])
        
        # Terms with other facilities k
        # For k not in {i,j}:
        # a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]  : delta += a[i,k]*(b[lj,p[k]] - b[li,p[k]])
        # a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]  : delta += a[k,i]*(b[p[k],lj] - b[p[k],li])
        # a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]  : delta += a[j,k]*(b[li,p[k]] - b[lj,p[k]])
        # a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]  : delta += a[k,j]*(b[p[k],li] - b[p[k],lj])
        
        # Vectorized computation for all k:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            
            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))
        
        return delta
    
    # Local search: iterate through all pairs, make improving swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                delta = compute_swap_delta(i, j, p)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    if best_cost < best_cost:
                        best_p = p.copy()
                        best_cost = compute_cost_fast(p)  # Recalculate to avoid drift
    
    # Try multiple restarts with different initial heuristics if time permits
    if time.time() < deadline - 5:
        # Try a different initial heuristic: random with local search
        rng = np.random.RandomState(42)
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Different initial permutations
            if restart == 0:
                # Inverse rank matching
                heavy2 = np.argsort(a.sum(axis=1), kind="stable")
                central2 = np.argsort(-b.sum(axis=1), kind="stable")
                p2 = np.empty(n, dtype=np.int64)
                p2[heavy2] = central2
            elif restart == 1:
                # Random
                p2 = rng.permutation(n).astype(np.int64)
            else:
                # Based on column sums
                heavy3 = np.argsort(-a.sum(axis=0), kind="stable")
                central3 = np.argsort(b.sum(axis=0), kind="stable")
                p2 = np.empty(n, dtype=np.int64)
                p2[heavy3] = central3
            
            p2 = p2.astype(np.int64)
            
            # Local search on this start
            cost2 = compute_cost_fast(p2)
            
            improved2 = True
            while improved2 and time.time() < deadline - 5:
                improved2 = False
                for i in range(n):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            break
                        delta = compute_swap_delta(i, j, p2)
                        if delta < 0:
                            p2[i], p2[j] = p2[j], p2[i]
                            cost2 += delta
                            improved2 = True
                
                # Recompute cost periodically to avoid drift
                if not improved2:
                    cost2 = compute_cost_fast(p2)
            
            if cost2 < best_cost:
                best_p = p2.copy()
                best_cost = cost2
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
