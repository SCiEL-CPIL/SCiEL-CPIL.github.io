"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Solve the linear relaxation
    # Objective approx: sum_i (sum_j a[i,j]) * (sum_k b[p[i], p[k]])
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Cost matrix for LAP: C[i, j] = a_row_sums[i] * b_row_sums[j]
    # We want to minimize this.
    # linear_sum_assignment minimizes the sum of C[i, p[i]].
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    # Note: a and b are int64, but LAP works with floats/ints.
    _, p = linear_sum_assignment(cost_matrix)
    p = p.astype(np.int64)
    
    # 2. Local Search with efficient delta calculation
    # Precompute permuted B for fast access? 
    # We will compute deltas on the fly using vectorized operations.
    
    def compute_cost(p):
        # sum_i sum_j a[i,j] * b[p[i], p[j]]
        # Vectorized: sum_i (a[i] @ b[p[i], p])
        # b[p[i], p] is a row of B permuted by p.
        # We can do: sum_i np.dot(a[i], b[p[i], p])
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    current_cost = compute_cost(p)
    
    # Precompute row sums of A for potential use, but we need full rows for deltas.
    
    improved = True
    max_passes = 20
    pass_count = 0
    
    # To speed up, we can precompute B rows indexed by p?
    # b_perm[i, k] = b[p[i], p[k]]
    # Then R_i = sum_k a[i, k] * b_perm[i, k]
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Precompute b_perm for this pass?
        # b_perm[i, k] = b[p[i], p[k]]
        # This is O(n^2) memory and time.
        b_perm = b[np.ix_(p, p)]
        
        for i in range(n):
            for j in range(i + 1, n):
                u, v = p[i], p[j]
                
                # Calculate delta for swapping p[i] and p[j]
                # Old R_i = sum_k a[i, k] * b[u, p[k]]
                # New R_i = sum_k a[i, k] * b[v, p[k]]
                # Diff R_i = sum_k a[i, k] * (b[u, p[k]] - b[v, p[k]])
                
                # We can use b_perm to get b[u, p[k]]?
                # b_perm[i, k] is b[p[i], p[k]] = b[u, p[k]].
                # So Old R_i = a[i] @ b_perm[i]
                
                # For New R_i, we need b[v, p[k]].
                # This is not directly in b_perm unless we construct it.
                # However, we can compute the difference vector.
                
                # Let's stick to the O(n) dot product method for simplicity and correctness first.
                # delta_R_i = np.dot(a[i], b[u, p] - b[v, p])
                # delta_R_j = np.dot(a[j], b[v, p] - b[u, p])
                
                # Similarly for C (column sums):
                # Old C_i = sum_k a[k, i] * b[p[k], u]
                # New C_i = sum_k a[k, i] * b[p[k], v]
                # delta_C_i = np.dot(a[:, i], b[p, u] - b[p, v])
                
                # Note: b[p, u] is a column of b indexed by p.
                
                diff_b_rows = b[u, p] - b[v, p]
                delta_R_i = np.dot(a[i], diff_b_rows)
                delta_R_j = np.dot(a[j], -diff_b_rows) # b[v,p] - b[u,p]
                
                diff_b_cols = b[p, u] - b[p, v]
                delta_C_i = np.dot(a[:, i], diff_b_cols)
                delta_C_j = np.dot(a[:, j], -diff_b_cols)
                
                delta_cost = delta_R_i + delta_R_j + delta_C_i + delta_C_j
                
                if delta_cost < 0:
                    p[i], p[j] = v, u
                    current_cost += delta_cost
                    improved = True
                    # Note: b_perm is now stale, but we don't use it in the delta calc above.
                    # The delta calc uses b[u, p] and b[p, u] which are updated by p swap.
                    # So the next iteration's delta calc will be correct.
    
    return p.tolist()
# EVOLVE-BLOCK-END
