import numpy as np
import time

def solve(a, b):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = len(a)
    
    # 1. Spectral Initialization
    # Compute eigenvectors
    # Note: For symmetric matrices, eigenvectors are orthogonal.
    # QAPLIB matrices are usually symmetric.
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
    except Exception:
        # Fallback to rank matching if eig fails
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()

    # Use the principal eigenvector (largest eigenvalue)
    # eigh returns eigenvalues in ascending order, so last is largest
    v_a = eigvecs_a[:, -1]
    v_b = eigvecs_b[:, -1]
    
    # Align signs to maximize correlation (or minimize, depending on interpretation)
    # If we want to match "high" in A with "high" in B, we should sort both descending.
    # But we need to check if reversing one helps.
    # Standard heuristic: sort facilities by v_a, locations by v_b.
    
    order_a = np.argsort(-v_a, kind="stable")
    order_b = np.argsort(-v_b, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[order_a] = order_b
    
    # Alternative spectral init: reverse one
    p_rev = np.empty(n, dtype=np.int64)
    p_rev[order_a] = order_b[::-1]
    
    # Compute cost for both and pick better
    def compute_cost(p):
        # sum a[i,j] * b[p[i], p[j]]
        # Vectorized: 
        # Create matrix B_p where B_p[i,j] = b[p[i], p[j]]
        # Then sum a * B_p
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)
        
    cost_p = compute_cost(p)
    cost_pr = compute_cost(p_rev)
    
    if cost_pr < cost_p:
        p = p_rev
        best_cost = cost_pr
    else:
        best_cost = cost_p
        
    # Local Search Function
    def local_search(p, current_cost, time_limit):
        p = p.copy()
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Compute deltas for all pairs
            # M = a @ b[p, :]
            B_p = b[p, :] # Shape (n, n), row i is b[p[i], :]
            M = a @ B_p   # Shape (n, n)
            
            # S_ij = M[i,j] - M[i,i] - M[j,j] + M[j,i]
            # We need to compute this for all i, j
            # M_ii is diag(M)
            M_ii = np.diag(M)
            # S = M - M_ii[:, None] - M_ii[None, :] + M.T
            S = M - M_ii[:, np.newaxis] - M_ii[np.newaxis, :] + M.T
            
            # Correction term: (a_ii + a_jj - a_ji - a_ij) * b[p_i, p_j]
            # D_ij = b[p[i], p[j]]
            D = B_p # D[i,j] is b[p[i], p[j]]
            
            # A_ii = np.diag(a)
            A_ii = np.diag(a)
            # Term = (A_ii[:, None] + A_ii[None, :] - a.T - a) * D
            # Note: a.T[i,j] = a[j,i]
            # a[i,j] is a_ij
            # We need a_ji which is a[j,i] = a.T[i,j]
            # We need a_ij which is a[i,j] = a[i,j]
            
            # Correction = (a_ii + a_jj - a_ji - a_ij) * D_ij
            # a_ii is A_ii[i]
            # a_jj is A_ii[j]
            # a_ji is a.T[i,j]
            # a_ij is a[i,j]
            
            corr = (A_ii[:, np.newaxis] + A_ii[np.newaxis, :] - a.T - a) * D
            
            Delta = S - corr
            
            # Set diagonal to inf to avoid self-swaps
            np.fill_diagonal(Delta, np.inf)
            
            # Find best swap
            min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
            i, j = min_idx
            min_delta = Delta[i, j]
            
            if min_delta < 0:
                # Apply swap
                p[i], p[j] = p[j], p[i]
                current_cost += min_delta
                improved = True
                
        return p, current_cost

    # ILS Loop
    best_p = p.copy()
    
    while time.time() < deadline:
        # Local Search
        p_ls, cost_ls = local_search(p, best_cost, deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Perturbation
        # Randomly swap 5-10 pairs
        num_swaps = np.random.randint(5, 15)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            
    return best_p.tolist()
