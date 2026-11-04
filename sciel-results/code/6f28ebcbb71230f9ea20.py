"""QAP solver for tho150 using Frank-Wolfe over the Birkhoff polytope + local search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(p, a, b):
    """Compute QAP cost for permutation p: sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _fw_iteration(X, G, max_iter_hungarian=1):
    """
    One Frank-Wolfe step: solve linear subproblem min <G, Z> over Birkhoff polytope
    using Hungarian algorithm, then update X = (1-tau)*X + tau*Z.
    Returns updated X and the step size tau.
    """
    # Linear subproblem: min sum_{i,j} G[i,j] * Z[i,j] s.t. Z in Birkhoff
    row_ind, col_ind = linear_sum_assignment(G)
    Z = np.zeros_like(G)
    Z[row_ind, col_ind] = 1.0

    # Step size: full step (tau=1) for simplicity, or line search
    # Use standard FW step size: tau_k = 2/(k+2), but we'll use a fixed small step
    # Actually, let's use the standard FW step size schedule
    # We'll pass in iteration count externally; here use tau=1.0 (full step)
    # Better: use tau = 2/(t+2) where t is iteration index
    # For now, use a simple approach: full step
    tau = 1.0
    X_new = (1 - tau) * X + tau * Z
    return X_new, tau


def _round_to_permutation(X):
    """Round a doubly-stochastic matrix to a permutation using Hungarian."""
    row_ind, col_ind = linear_sum_assignment(-X)  # maximize sum X[i,j]
    p = np.empty(len(X), dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _local_search_2swap(p, a, b, max_iter=5000, time_limit=5.0):
    """
    Local search with 2-swap moves.
    p: permutation array (facility i -> location p[i])
    Returns improved permutation.
    """
    n = len(p)
    p = p.copy()
    cost = _cost(p, a, b)
    start_time = time.time()
    
    # Precompute: for each pair (i,j), the delta of swapping p[i] and p[j]
    # delta(i,j) = sum_k [a[i,k]*b[p[j],p[k]] + a[j,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[j]] + a[k,j]*b[p[k],p[i]]]
    #            - sum_k [a[i,k]*b[p[i],p[k]] + a[j,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[i]] + a[k,j]*b[p[k],p[j]]]
    # Simplified: delta = (a[i,j]+a[j,i])* (b[p[i],p[j]] - b[p[j],p[i]])
    #            + sum_{k!=i,j} [a[i,k]+a[k,i]+a[j,k]+a[k,j]] * (b[p[i],p[k]] - b[p[j],p[k]])
    #            + sum_{k!=i,j} [a[i,k]+a[k,i]+a[j,k]+a[k,j]] * (b[p[k],p[i]] - b[p[k],p[j]])
    # This is O(n) per pair, O(n^3) total. For n=150, that's 3.3M ops per full pass.
    
    # For efficiency, precompute row sums of a and b
    # Actually, let's just do a simpler O(n) delta computation per pair
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    return p, cost
                # Compute delta for swapping p[i] and p[j]
                pi = p[i]
                pj = p[j]
                if pi == pj:
                    continue
                # delta = sum_k [a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] + a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi]]
                #       - sum_k [a[i,k]*b[pi,p[k]] + a[j,k]*b[pj,p[k]] + a[k,i]*b[p[k],pi] + a[k,j]*b[p[k],pj]]
                # = sum_k (a[i,k]+a[k,i]+a[j,k]+a[k,j]) * (b[pj,p[k]] - b[pi,p[k]])  -- wait, this isn't right
                # Let me re-derive:
                # Original contribution involving i and j:
                #   sum_k a[i,k]*b[pi,p[k]] + sum_k a[j,k]*b[pj,p[k]]
                #   + sum_k a[k,i]*b[p[k],pi] + sum_k a[k,j]*b[p[k],pj]
                #   + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]  (these are included in the sums above when k=j and k=i)
                # After swap:
                #   sum_k a[i,k]*b[pj,p[k]] + sum_k a[j,k]*b[pi,p[k]]
                #   + sum_k a[k,i]*b[p[k],pj] + sum_k a[k,j]*b[p[k],pi]
                # Delta = [sum_k a[i,k]*(b[pj,p[k]]-b[pi,p[k]]) + sum_k a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
                #        + sum_k a[k,i]*(b[p[k],pj]-b[p[k],pi]) + sum_k a[k,j]*(b[p[k],pi]-b[p[k],pj])]
                # = sum_k (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
                #   + sum_k (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
                
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    delta += (a[i,k] - a[j,k]) * (b[pj,k] - b[pi,k])
                    delta += (a[k,i] - a[k,j]) * (b[k,pj] - b[k,pi])
                # Add the i-j and j-i terms
                delta += a[i,j] * (b[pj,pj] - b[pi,pj])  # a[i,j]*b[pj,pj] - a[i,j]*b[pi,pj]... wait
                # When k=j: a[i,j]*b[pj,p[j]] becomes a[i,j]*b[pj,pj] after swap? No.
                # Let me be more careful. The full sum includes all k.
                # For k=j: original has a[i,j]*b[pi,pj], new has a[i,j]*b[pj,pj]
                # For k=i: original has a[i,i]*b[pi,pi], new has a[i,i]*b[pj,pi]
                # These are handled in the general formula if we don't skip i and j.
                # Let me redo without skipping:
                delta = 0
                for k in range(n):
                    delta += (a[i,k] - a[j,k]) * (b[pj,k] - b[pi,k])
                    delta += (a[k,i] - a[k,j]) * (b[k,pj] - b[k,pi])
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
    
    return p, cost


def solve(a, b):
    n = len(a)
    overall_start = time.time()
    deadline = overall_start + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Frank-Wolfe on the Birkhoff polytope
    # Initialize X as uniform doubly-stochastic matrix
    X = np.full((n, n), 1.0 / n)
    
    best_p = None
    best_cost = np.inf
    
    # Get initial permutation from rank-matching as a baseline
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    best_p = p_init.copy()
    best_cost = _cost(best_p, a, b)
    
    # Frank-Wolfe iterations
    fw_iter = 0
    max_fw_iters = 200
    while fw_iter < max_fw_iters and (time.time() - overall_start) < deadline - 60.0:
        # Compute gradient of QAP objective w.r.t. X
        # f(X) = sum_{i,j,k,l} a[i,j]*b[k,l]*X[i,k]*X[j,l]
        # grad[i,k] = sum_{j,l} a[i,j]*b[k,l]*X[j,l] + sum_{j,l} a[j,i]*b[l,k]*X[j,l]
        #          = (a @ X @ b.T)[i,k] + (a.T @ X @ b)[i,k]
        # Wait, let me re-derive.
        # f(X) = sum_{i,j} a[i,j] * sum_{k,l} b[k,l] * X[i,k] * X[j,l]
        #      = sum_{i,j,k,l} a[i,j] * b[k,l] * X[i,k] * X[j,l]
        # grad[i,k] = sum_{j,l} a[i,j] * b[k,l] * X[j,l] + sum_{j,l} a[j,i] * b[l,k] * X[j,l]
        #          = (a @ X @ b.T)[i,k] + (a.T @ X @ b)[i,k]
        
        G = a @ X @ b.T + a.T @ X @ b
        
        # Frank-Wolfe step
        X, tau = _fw_iteration(X, G)
        fw_iter += 1
        
        # Round to permutation and evaluate
        p_round = _round_to_permutation(X)
        c_round = _cost(p_round, a, b)
        if c_round < best_cost:
            best_cost = c_round
            best_p = p_round.copy()
    
    # Phase 2: Local search polish from the best solution
    if (time.time() - overall_start) < deadline - 10.0:
        p_ls, cost_ls = _local_search_2swap(best_p, a, b, max_iter=10000, time_limit=10.0)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Phase 3: If time remains, do another round of local search from a different start
    if (time.time() - overall_start) < deadline - 15.0:
        # Try local search from the rank-matching start too
        p_ls2, cost_ls2 = _local_search_2swap(p_init, a, b, max_iter=10000, time_limit=10.0)
        if cost_ls2 < best_cost:
            best_cost = cost_ls2
            best_p = p_ls2
    
    # Final check: make sure we return a valid permutation
    assert len(best_p) == n
    assert set(best_p.tolist()) == set(range(n)), "Invalid permutation"
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
