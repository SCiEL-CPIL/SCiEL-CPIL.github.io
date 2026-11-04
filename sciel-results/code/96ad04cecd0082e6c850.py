"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Rank matching
    # High flow sums -> Low distance sums
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    # Sort facilities by descending flow sum
    # Sort locations by ascending distance sum
    heavy = np.argsort(-a_sums, kind="stable")
    central = np.argsort(b_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = float('inf')
    
    def calc_cost(p):
        # C = sum a[i, j] * b[p[i], p[j]]
        # Vectorized: (a * b[p[:, None], p[None, :]]).sum()
        # This creates an n x n temporary array. For n=256, 256*256*8 bytes = 512KB. Fine.
        return np.sum(a * b[p[:, None], p[None, :]])
    
    current_cost = calc_cost(p)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    # Helper to perform best improvement 2-opt
    def local_search(p, current_cost):
        nonlocal best_p, best_cost
        max_iterations = 10000
        for _ in range(max_iterations):
            if time.time() > deadline:
                return p, current_cost
            
            # Compute S = M_row + N
            # M_row[i, l] = sum_k a[i, k] b[l, p[k]]
            # B_perm_cols: columns of b permuted by p
            # b_perm_cols[:, k] = b[:, p[k]]
            b_perm_cols = b[:, p]
            m_row = a @ b_perm_cols
            
            # N[i, l] = sum_k a[k, i] b[p[k], l]
            # b_perm_rows: rows of b permuted by p
            # b_perm_rows[k, :] = b[p[k], :]
            b_perm_rows = b[p, :]
            n_mat = a.T @ b_perm_rows
            
            s = m_row + n_mat
            
            # Compute Delta matrix
            # Delta[i, j] = S[i, p[j]] + S[j, p[i]] - S[i, p[i]] - S[j, p[j]] 
            #              - a[i, j] * b[p[j], p[j]] + a[i, j] * b[p[j], p[i]]
            
            s_col = s[:, p]      # s_col[i, j] = s[i, p[j]]
            s_row = s[p, :]      # s_row[i, j] = s[p[i], j] -> Wait, s_row[i, j] should be s[j, p[i]]
            # Let's check indices.
            # We want s[j, p[i]] for each i, j.
            # s_row[i, j] = s[p[i], j] is NOT s[j, p[i]].
            # We need s[j, p[i]].
            # Let s_trans = s.T. Then s_trans[p[i], j] = s[j, p[i]].
            # So s_row_correct = s.T[p, :]? No.
            # s[j, p[i]]: fix i, vary j.
            # This is column p[i] of s, transposed?
            # s[:, p[i]] is a vector of s[k, p[i]] for k=0..n-1.
            # We want s[j, p[i]] for j=0..n-1.
            # So s_row[i, j] = s[j, p[i]].
            # This is s.T[p, :]?
            # s.T[p, j] = s[j, p]. No.
            # s.T[p[i], j] = s[j, p[i]].
            # So s_row = s.T[p, :] gives s_row[i, j] = s[j, p[i]]. Correct.
            
            s_row = s.T[p, :]
            
            s_diag_u = s[np.arange(n), p]      # s[i, p[i]]
            s_diag_v = s[np.arange(n), p]      # s[j, p[j]]
            
            # Term: -a[i, j] * b[p[j], p[j]]
            b_diag_p = b[p, p]                 # b_diag_p[j] = b[p[j], p[j]]
            term1 = a * b_diag_p[None, :]
            
            # Term: +a[i, j] * b[p[j], p[i]]
            b_cross = b[p[:, None], p[None, :]] # b_cross[i, j] = b[p[i], p[j]]
            # We need b[p[j], p[i]]. This is b_cross.T.
            b_cross_t = b_cross.T
            
            term2 = a * b_cross_t
            
            delta = s_col + s_row - s_diag_u[:, None] - s_diag_v[None, :] - term1 + term2
            
            # Find min delta
            # We only consider i < j or i != j? Swapping i and j is same as j and i.
            # Delta is symmetric?
            # Delta[i, j] vs Delta[j, i].
            # s_col[i, j] = s[i, p[j]]. s_col[j, i] = s[j, p[i]].
            # s_row[i, j] = s[j, p[i]]. s_row[j, i] = s[i, p[j]].
            # So s_col + s_row is symmetric.
            # s_diag_u[:, None] - s_diag_v[None, :] is NOT symmetric.
            # s_diag_u[i] - s_diag_v[j] vs s_diag_u[j] - s_diag_v[i].
            # term1: a[i, j] b[p[j], p[j]]. term1[j, i] = a[j, i] b[p[i], p[i]].
            # term2: a[i, j] b[p[j], p[i]]. term2[j, i] = a[j, i] b[p[i], p[j]].
            # Since a and b are symmetric (QAP matrices usually are, but tai256c might not be? 
            # QAPLIB instances usually have symmetric A and B. Let's assume they are symmetric 
            # but handle non-symmetric just in case by using the formula as derived).
            # Actually, the cost function sum a[i,j] b[p[i], p[j]] is not necessarily symmetric in i,j 
            # if A or B are not symmetric. But tai256c has symmetric A and B.
            # If A and B are symmetric, Delta should be symmetric.
            
            # Find the index of the minimum delta
            # We should ignore diagonal (i=j) as swapping with self is 0.
            # Set diagonal to inf.
            np.fill_diagonal(delta, np.inf)
            
            min_idx = np.unravel_index(np.argmin(delta), delta.shape)
            min_delta = delta[min_idx]
            
            if min_delta < -1e-9: # Improvement
                i, j = min_idx
                # Swap p[i] and p[j]
                p[i], p[j] = p[j], p[i]
                current_cost += min_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                break
                
        return p, current_cost

    # Run local search
    p, current_cost = local_search(p, current_cost)
    
    # If time remains, do restarts with perturbations
    # We can perturb by random 2-swaps or segment reversals
    while time.time() < deadline - 5.0:
        # Perturb
        p = best_p.copy()
        # Random 2-swap
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
        
        current_cost = calc_cost(p)
        
        p, current_cost = local_search(p, current_cost)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
