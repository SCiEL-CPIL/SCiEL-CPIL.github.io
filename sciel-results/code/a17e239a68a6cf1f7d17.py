"""Frank-Wolfe / Birkhoff relaxation for QAP tai256c, with spectral init and fast 2-opt polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh


def _qap_cost(a, b, p):
    """Compute the exact integer QAP cost for permutation p."""
    n = len(p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _spectral_init(a, b):
    """
    Generate a high-quality initial permutation using spectral embedding.
    We construct a matrix M that captures the interaction structure and use
    its leading eigenvectors to assign facilities to locations.
    """
    n = len(a)
    
    # Construct the linearization matrix M
    # M = diag(a @ 1) + elementwise(a * b) is one option, but let's try:
    # The QAP cost can be approximated by trace-based formulations.
    # A common spectral approach uses the matrix:
    # M = A * B (Hadamard product) + diag(A.sum(axis=1))
    # Or more commonly, we solve a relaxed problem.
    # Let's use the matrix: M = A + B (if dimensions match and we treat them similarly)
    # Actually, for QAP, a better spectral init uses the eigenvectors of A and B.
    # Let's try: Project facilities onto the space of locations using B's eigenvectors.
    
    # Simple spectral init:
    # 1. Compute leading eigenvectors of B (distance matrix)
    # 2. Assign facilities based on their "importance" (row sums of A) to locations
    #    that align with the principal directions of B.
    
    # For symmetric B, we can use eigsh to get the largest eigenvectors.
    # We want to assign heavy facilities (large row sum in A) to "central" locations.
    # In spectral clustering/embedding, central locations are often those with
    # small distances to others, or high eigenvalues.
    
    # Let's try a different approach:
    # Solve min trace(X^T A X B) approximately.
    # This is equivalent to finding X that aligns A and B.
    # A simple heuristic: use the eigenvectors of A and B.
    
    # Compute top k eigenvectors of A and B
    k = min(16, n - 1)
    
    try:
        # For symmetric matrices, use eigsh
        if np.allclose(a, a.T) and np.allclose(b, b.T):
            # Get largest eigenvalues/vectors of A
            eigvals_a, eigvecs_a = eigsh(a, k=k, which='LM')
            # Get largest eigenvalues/vectors of B
            eigvals_b, eigvecs_b = eigsh(b, k=k, which='LM')
            
            # Create embeddings
            # Facilities are embedded in the space of A's eigenvectors
            # Locations are embedded in the space of B's eigenvectors
            # We want to match them.
            
            # Simple approach: sort facilities by projection onto first eigenvector of A
            # and sort locations by projection onto first eigenvector of B
            # Then assign in order (or reverse order depending on signs)
            
            # Project facilities onto first few eigenvectors of A
            fac_proj = eigvecs_a @ np.diag(eigvals_a)  # Weighted projection
            loc_proj = eigvecs_b @ np.diag(eigvals_b)
            
            # Sort by the first principal component
            fac_order = np.argsort(-fac_proj[:, 0])
            loc_order = np.argsort(-loc_proj[:, 0])
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p.tolist()
            
    except Exception:
        pass
    
    # Fallback: greedy rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.tolist()


def _frank_wolfe_qap(a, b, init_perm, max_iters=300, time_budget=100.0):
    """
    Frank-Wolfe (conditional gradient) on the Birkhoff polytope.
    """
    n = len(a)
    t_start = time.time()
    
    # Initialize X as the permutation matrix of init_perm
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), init_perm] = 1.0
    
    best_cost = _qap_cost(a, b, init_perm)
    best_perm = list(init_perm)
    
    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > time_budget:
            break
        
        # Compute gradient
        AX = a @ X
        if np.allclose(a, a.T) and np.allclose(b, b.T):
            grad = 2.0 * (AX @ b)
        else:
            ATX = a.T @ X
            grad = (AX @ b) + (ATX @ b.T)
        
        # Solve linear assignment on grad to get next vertex
        row_ind, col_ind = linear_sum_assignment(grad)
        V = np.zeros((n, n), dtype=np.float64)
        V[row_ind, col_ind] = 1.0
        
        # Line search
        gamma = 2.0 / (it + 2.0)
        
        X_new = (1.0 - gamma) * X + gamma * V
        X = X_new
        
        # Periodically extract a permutation and evaluate
        if it % 5 == 0:
            r_idx, c_idx = linear_sum_assignment(-X)
            perm = np.empty(n, dtype=np.int64)
            perm[r_idx] = c_idx
            cost = _qap_cost(a, b, perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        
        # Also try the current vertex V as a permutation
        perm_v = np.empty(n, dtype=np.int64)
        perm_v[row_ind] = col_ind
        cost_v = _qap_cost(a, b, perm_v)
        if cost_v < best_cost:
            best_cost = cost_v
            best_perm = list(perm_v)
    
    return best_perm, best_cost


def _local_search_2opt_fast(a, b, p, time_budget=20.0):
    """
    Fast 2-opt local search with O(n) incremental cost delta for swaps.
    """
    n = len(p)
    p = list(p)
    best_cost = _qap_cost(a, b, p)
    t_start = time.time()
    
    # Precompute row sums of a for quick delta computation
    a_row_sums = a.sum(axis=1)
    
    improved = True
    passes = 0
    max_passes = 20
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() - t_start > time_budget:
            break
        
        for i in range(n):
            if time.time() - t_start > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - t_start > time_budget:
                    break
                
                # Compute delta for swapping p[i] and p[j]
                # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])
                #       + a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])
                #       + a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])
                
                pi = p[i]
                pj = p[j]
                
                # For symmetric b, the direct term is zero
                # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pi,p[k]] - b[pj,p[k]])
                
                # Vectorized computation
                diff_a = a[i] - a[j]
                diff_b = b[pi, p] - b[pj, p]
                
                # Exclude k=i and k=j
                diff_b[i] = 0
                diff_b[j] = 0
                
                delta = np.dot(diff_a, diff_b)
                
                # Add direct interaction if b is not symmetric
                if not np.allclose(b, b.T):
                    delta += a[i, j] * (b[pi, pj] - b[pj, pi])
                    delta += a[j, i] * (b[pj, pi] - b[pi, pj])
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += int(delta)
                    improved = True
    
    return p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    total_time_budget = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Spectral initialization
    p_init = _spectral_init(a, b)
    
    # Phase 2: Frank-Wolfe relaxation
    fw_time_budget = 80.0
    p_fw, cost_fw = _frank_wolfe_qap(a, b, p_init, max_iters=500, time_budget=fw_time_budget)
    
    elapsed = time.time() - t_start
    remaining = total_time_budget - elapsed
    
    if remaining < 5.0:
        return p_fw
    
    # Phase 3: Fast local search polish
    ls_time_budget = min(remaining - 2.0, 40.0)
    p_ls, cost_ls = _local_search_2opt_fast(a, b, p_fw, time_budget=ls_time_budget)
    
    return p_ls
# EVOLVE-BLOCK-END
