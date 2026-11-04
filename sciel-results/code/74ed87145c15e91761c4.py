import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution via Linear Assignment Problem
    # Use the linearized objective sum_k a[i, k] * b[j, k] as the cost matrix.
    # This corresponds to M = A @ B.T.
    # This captures the pairwise affinity between facility i's flow profile and location j's distance profile.
    M = a @ b.T
    
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(M)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # 2. Local Search
    # Precompute S[i, v] = sum_y a[i, y] * b[v, p[y]]
    # S = A @ B[:, p]
    B_perm = b[:, p]
    S = a @ B_perm
    
    # Current cost
    current_cost = np.sum(S[np.arange(n), p])
    
    # Local search loop
    # We use a best-improvement strategy
    # To avoid O(n^3) per pass being too slow in pure Python loops, 
    # we rely on NumPy vectorization for the inner sum.
    
    for _ in range(100): # Limit passes to ensure time limit
        best_delta = np.inf
        best_i, best_k = -1, -1
        
        # Vectorized evaluation of all pairs is hard due to the mask.
        # We iterate over pairs but use NumPy for the dot product part.
        # n=150, pairs=11k. 11k * O(n) dot product.
        
        # Optimization: Precompute A_col_diffs? No, depends on i,k.
        
        for i in range(n):
            for k in range(i + 1, n):
                u = p[i]
                v = p[k]
                
                # V_diff[w] = b[w, v] - b[w, u]
                V_diff = b[:, v] - b[:, u]
                
                # vec[j] = a[j, i] - a[j, k]
                vec = a[:, i] - a[:, k]
                
                # Mask for j != i, k
                # Instead of masking, we can compute full dot and subtract terms for i, k
                # sum_all = dot(vec, V_diff[p])
                # term_i = vec[i] * V_diff[p[i]] = (a[i,i]-a[i,k]) * (b[u,v]-b[u,u])
                # term_k = vec[k] * V_diff[p[k]] = (a[k,i]-a[k,k]) * (b[v,v]-b[v,u])
                # sum_masked = sum_all - term_i - term_k
                
                sum_all = np.dot(vec, V_diff[p])
                term_i = (a[i, i] - a[i, k]) * (b[u, v] - b[u, u])
                term_k = (a[k, i] - a[k, k]) * (b[v, v] - b[v, u])
                sum_part = sum_all - term_i - term_k
                
                # Delta calculation
                # Delta = (S'_i,v - S_i,u) + (S'_k,u - S_k,v) + sum_part
                # S'_i,v = S_i,v + (a[i,i]-a[i,k]) * (b[v,v]-b[v,u])
                # S'_k,u = S_k,u + (a[k,i]-a[k,k]) * (b[u,v]-b[u,u])
                
                delta_1 = S[i, v] + (a[i, i] - a[i, k]) * (b[v, v] - b[v, u]) - S[i, u]
                delta_2 = S[k, u] + (a[k, i] - a[k, k]) * (b[u, v] - b[u, u]) - S[k, v]
                
                delta = delta_1 + delta_2 + sum_part
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_k = k
        
        if best_delta >= 0 or best_i == -1:
            break
        
        # Apply swap
        i, k = best_i, best_k
        u, v = p[i], p[k]
        
        p[i], p[k] = v, u
        
        # Update S
        # S[:, w] += (A[:, i] - A[:, k]) * (B[w, v] - B[w, u])
        # S += outer(A[:, i] - A[:, k], B[:, v] - B[:, u])
        
        # Note: B[:, v] - B[:, u] is a vector of length n (for index w)
        # A[:, i] - A[:, k] is a vector of length n (for index j)
        # S[j, w] += (A[j, i] - A[j, k]) * (B[w, v] - B[w, u])
        
        diff_A = a[:, i] - a[:, k]
        diff_B = b[:, v] - b[:, u]
        
        S += np.outer(diff_A, diff_B)
        
        current_cost += best_delta

    return p.tolist()
