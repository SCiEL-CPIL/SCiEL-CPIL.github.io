"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    t0 = time.time()
    time_limit = 115.0  # leave 5s buffer for overhead

    # Convert to float for optimization
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def perm_cost(p):
        """Compute cost of permutation p."""
        return int(np.sum(a_f * b_f[np.ix_(p, p)]))

    # Sinkhorn-Knopp for projection to doubly stochastic
    def sinkhorn(X, max_iter=50, tol=1e-6):
        """Project X to doubly stochastic matrix using Sinkhorn-Knopp."""
        X = np.maximum(X, 1e-12)
        for _ in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            if np.max(np.abs(X.sum(axis=1) - 1.0)) < tol and np.max(np.abs(X.sum(axis=0) - 1.0)) < tol:
                break
        return X

    # Correct local search: compute delta cost for swapping p[i] and p[j]
    def swap_delta(p, i, j):
        """Compute cost change from swapping p[i] and p[j]."""
        pi, pj = p[i], p[j]
        
        # All terms involving i or j as first index:
        # a[i,k]*b[pi,p[k]] for all k, and a[j,k]*b[pj,p[k]] for all k
        # After swap: a[i,k]*b[pj,p[k]] and a[j,k]*b[pi,p[k]]
        # Delta from these: sum_k (a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]])
        
        # All terms involving i or j as second index:
        # a[k,i]*b[p[k],pi] for all k, and a[k,j]*b[p[k],pj] for all k
        # After swap: a[k,i]*b[p[k],pj] and a[k,j]*b[p[k],pi]
        # Delta from these: sum_k (a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi] - a[k,i]*b[p[k],pi] - a[k,j]*b[p[k],pj])
        
        # Combine: for k != i,j, we have contributions from both first and second index
        # For k = i: a[i,i]*b[pi,pi] + a[i,i]*b[pj,pj] (first index i, second index i and j)
        #            a[i,i]*b[pi,pi] + a[i,i]*b[pj,pj] (second index i, first index i and j)
        #            Actually need to be careful about double counting
        
        # Let me compute it directly by enumerating all affected terms:
        # Terms where first index is i: a[i,k]*b[pi,p[k]] for k=0..n-1
        #   After swap: a[i,k]*b[pj,p[k]]
        #   Delta_i_first = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
        
        # Terms where first index is j: a[j,k]*b[pj,p[k]] for k=0..n-1
        #   After swap: a[j,k]*b[pi,p[k]]
        #   Delta_j_first = sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
        
        # Terms where second index is i: a[k,i]*b[p[k],pi] for k=0..n-1
        #   After swap: a[k,i]*b[p[k],pj]
        #   Delta_i_second = sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi])
        
        # Terms where second index is j: a[k,j]*b[p[k],pj] for k=0..n-1
        #   After swap: a[k,j]*b[p[k],pi]
        #   Delta_j_second = sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
        
        # But wait - terms where both indices are i or j are counted twice!
        # Term a[i,i]*b[pi,pi]: counted in Delta_i_first (k=i) and Delta_i_second (k=i)
        # After swap: a[i,i]*b[pj,pj] - counted in both
        # So we need to subtract the double-counted terms
        
        # Actually, let's think about it differently. The total cost is:
        # C = sum_{i,j} a[i,j] * b[p[i],p[j]]
        
        # When we swap p[i] and p[j], the terms that change are:
        # 1. Terms where i is the first index: a[i,k]*b[pi,p[k]] -> a[i,k]*b[pj,p[k]] for all k
        # 2. Terms where j is the first index: a[j,k]*b[pj,p[k]] -> a[j,k]*b[pi,p[k]] for all k
        # 3. Terms where i is the second index: a[k,i]*b[p[k],pi] -> a[k,i]*b[p[k],pj] for all k
        # 4. Terms where j is the second index: a[k,j]*b[p[k],pj] -> a[k,j]*b[p[k],pi] for all k
        
        # But terms (i,i), (i,j), (j,i), (j,j) appear in multiple categories:
        # (i,i): appears in 1 (k=i) and 3 (k=i)
        # (i,j): appears in 1 (k=j) and 4 (k=i)
        # (j,i): appears in 2 (k=i) and 3 (k=j)
        # (j,j): appears in 2 (k=j) and 4 (k=j)
        
        # So the total delta is:
        delta = 0
        # Category 1: first index i
        for k in range(n):
            delta += a_f[i, k] * (b_f[pj, p[k]] - b_f[pi, p[k]])
        # Category 2: first index j
        for k in range(n):
            delta += a_f[j, k] * (b_f[pi, p[k]] - b_f[pj, p[k]])
        # Category 3: second index i
        for k in range(n):
            delta += a_f[k, i] * (b_f[p[k], pj] - b_f[p[k], pi])
        # Category 4: second index j
        for k in range(n):
            delta += a_f[k, j] * (b_f[p[k], pi] - b_f[p[k], pj])
        
        # Now subtract the double-counted terms:
        # (i,i) was counted twice (in cat 1 k=i and cat 3 k=i), should be counted once
        # The delta for (i,i) is: a[i,i]*(b[pj,pj]-b[pi,pi]) from cat1 + a[i,i]*(b[pj,pj]-b[pi,pi]) from cat3
        # But the actual change is just a[i,i]*(b[pj,pj]-b[pi,pi])
        # So we need to subtract one copy:
        delta -= a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi])
        
        # (j,j) was counted twice (in cat 2 k=j and cat 4 k=j), should be counted once
        delta -= a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])
        
        # (i,j) was counted in cat 1 (k=j) and cat 4 (k=i)
        # Cat 1 k=j: a[i,j]*(b[pj,pj]-b[pi,pj])
        # Cat 4 k=i: a[i,j]*(b[pi,pi]-b[pj,pi])
        # Actual change: a[i,j]*(b[pj,pi]-b[pi,pj])
        # Sum of cat1+cat4: a[i,j]*(b[pj,pj]-b[pi,pj]+b[pi,pi]-b[pj,pi])
        # We need: a[i,j]*(b[pj,pi]-b[pi,pj])
        # So subtract: a[i,j]*(b[pj,pj]-b[pi,pj]+b[pi,pi]-b[pj,pi] - b[pj,pi]+b[pi,pj])
        #            = a[i,j]*(b[pj,pj]+b[pi,pi]-2*b[pj,pi])
        delta -= a_f[i, j] * (b_f[pj, pj] + b_f[pi, pi] - 2 * b_f[pj, pi])
        
        # (j,i) was counted in cat 2 (k=i) and cat 3 (k=j)
        # Cat 2 k=i: a[j,i]*(b[pi,pi]-b[pj,pi])
        # Cat 3 k=j: a[j,i]*(b[pj,pj]-b[pi,pj])
        # Actual change: a[j,i]*(b[pi,pj]-b[pj,pi])
        # Sum: a[j,i]*(b[pi,pi]-b[pj,pi]+b[pj,pj]-b[pi,pj])
        # Need: a[j,i]*(b[pi,pj]-b[pj,pi])
        # Subtract: a[j,i]*(b[pi,pi]-b[pj,pi]+b[pj,pj]-b[pi,pj]-b[pi,pj]+b[pj,pi])
        #          = a[j,i]*(b[pi,pi]+b[pj,pj]-2*b[pi,pj])
        delta -= a_f[j, i] * (b_f[pi, pi] + b_f[pj, pj] - 2 * b_f[pi, pj])
        
        return delta

    def local_search_2opt(p, max_iter=100, time_limit_inner=15.0):
        """Improve permutation via pairwise swaps."""
        p = p.copy()
        current_cost = perm_cost(p)
        t_start = time.time()
        
        for _ in range(max_iter):
            if time.time() - t_start > time_limit_inner:
                break
            
            improved = False
            # Try random pairs of swaps
            for _ in range(200):
                i, j = np.random.choice(n, size=2, replace=False)
                if i == j:
                    continue
                
                delta = swap_delta(p, i, j)
                
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break
            
            if not improved:
                break
        
        return p, current_cost

    # Generate multiple initial permutations
    candidates = []
    
    # Strategy 1: Rank-matching (heavy facilities to central locations)
    heavy = np.argsort(-a_f.sum(axis=1), kind="stable")
    central = np.argsort(b_f.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    candidates.append(p1)
    
    # Strategy 2: Reverse rank-matching
    heavy2 = np.argsort(a_f.sum(axis=1), kind="stable")
    central2 = np.argsort(-b_f.sum(axis=1), kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    candidates.append(p2)
    
    # Strategy 3: Random
    p3 = np.random.permutation(n)
    candidates.append(p3)
    
    # Strategy 4: Greedy based on row sums
    row_sums_a = a_f.sum(axis=1)
    row_sums_b = b_f.sum(axis=1)
    p4 = np.argsort(np.argsort(row_sums_a, kind="stable"), kind="stable")
    p4 = np.empty(n, dtype=np.int64)
    # Map: facility with rank r in a gets location with rank r in b
    rank_a = np.argsort(row_sums_a, kind="stable")
    rank_b = np.argsort(row_sums_b, kind="stable")
    for fac, rank in enumerate(rank_a):
        p4[fac] = rank_b[rank]
    candidates.append(p4)
    
    best_cost = np.inf
    best_p = None
    
    for p_init in candidates:
        cost = perm_cost(p_init)
        if cost < best_cost:
            best_cost = cost
            best_p = p_init.copy()
    
    # Continuous relaxation for each good candidate
    for p_init in candidates[:2]:  # Only do relaxation for top 2
        if time.time() - t0 > time_limit - 30:
            break
        
        # Create initial doubly stochastic matrix
        X = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            X[i, p_init[i]] = 1.0
        
        # Gradient-based optimization with momentum
        grad = np.zeros((n, n), dtype=np.float64)
        step_size = 0.1
        momentum = 0.5
        
        best_relaxed_cost = np.inf
        best_X = X.copy()
        
        for iteration in range(500):
            if time.time() - t0 > time_limit - 20:
                break
            
            # Compute gradient: grad = A @ X @ B.T + B @ X.T @ A.T
            grad_new = a_f @ X @ b_f.T + b_f @ X.T @ a_f.T
            
            # Momentum update
            grad = momentum * grad + (1 - momentum) * grad_new
            
            # Compute current relaxed cost
            C = X.T @ a_f @ X
            cost = np.sum(b_f * C)
            
            if cost < best_relaxed_cost:
                best_relaxed_cost = cost
                best_X = X.copy()
            
            # Try to improve with backtracking line search
            improved = False
            alpha = step_size
            for _ in range(4):
                X_new = X - alpha * grad
                X_new = sinkhorn(X_new, max_iter=20)
                
                C_new = X_new.T @ a_f @ X_new
                cost_new = np.sum(b_f * C_new)
                
                if cost_new < cost:
                    X = X_new
                    step_size = min(alpha * 1.1, 1.0)
                    improved = True
                    break
                alpha *= 0.5
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-8:
                    break
        
        # Recover permutation from best_X
        row_ind, col_ind = linear_sum_assignment(-best_X)
        p_relaxed = np.empty(n, dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            p_relaxed[i] = j
        
        cost_relaxed = perm_cost(p_relaxed)
        if cost_relaxed < best_cost:
            best_cost = cost_relaxed
            best_p = p_relaxed.copy()
    
    # Apply local search to the best permutation
    if time.time() - t0 < time_limit - 10:
        p_ls, cost_ls = local_search_2opt(best_p, max_iter=100, time_limit_inner=15.0)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Final local search if time remains
    if time.time() - t0 < time_limit - 5:
        p_final, cost_final = local_search_2opt(best_p, max_iter=100, time_limit_inner=10.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
