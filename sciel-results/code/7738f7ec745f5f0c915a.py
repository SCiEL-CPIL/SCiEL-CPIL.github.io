"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """
    Solves the QAP instance using rank-matching initialization and 2-opt local search.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Initialization: Rank Matching
    # Heavy flows should be assigned to short distances.
    # Sort facilities by total flow (descending)
    flow_sums = a.sum(axis=1)
    # Sort locations by total distance (ascending)
    dist_sums = b.sum(axis=1)
    
    heavy_order = np.argsort(-flow_sums, kind='stable')
    central_order = np.argsort(dist_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Compute initial cost
    # Z = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Can be computed as trace(a @ b_perm) or sum(a * b[p[:,None], p[None,:]])
    # b_perm[i,j] = b[p[i], p[j]]
    # Efficient: np.sum(a * b[np.ix_(p, p)])
    current_cost = np.sum(a * b[np.ix_(p, p)])
    
    # 2. Local Search: 2-Opt (Pairwise Swaps)
    # We look for the best swap (i, j) that reduces cost.
    # Delta for swapping i and j (locations u and v):
    # Delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Note: Symmetry a[i,k]=a[k,i], b[u,v]=b[v,u].
    
    while time.time() < deadline:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Iterate over all pairs i < j
        for i in range(n):
            u = p[i]
            # Vectorized calculation of deltas for all j > i
            # We need: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v_j, p[k]] - b[u, p[k]])
            # This is tricky to fully vectorize over j because v_j changes.
            # Let's stick to the inner loop over j, but use NumPy for the sum over k.
            
            a_i = a[i]
            b_u = b[u] # b[u, :]
            
            for j in range(i + 1, n):
                v = p[j]
                a_j = a[j]
                b_v = b[v] # b[v, :]
                
                # diff_a = a_i - a_j
                # diff_b = b_v[p] - b_u[p]  -> b_v[p[0]], b_v[p[1]], ...
                # We need sum_{k != i,j} diff_a[k] * diff_b[k]
                
                # Construct diff_b vectorized
                # b_v_p = b_v[p]
                # b_u_p = b_u[p]
                # diff_b = b_v_p - b_u_p
                
                # We can compute the dot product of (a_i - a_j) and (b_v[p] - b_u[p])
                # Then subtract the terms for k=i and k=j.
                
                # term_all = np.dot(a_i - a_j, b_v[p] - b_u[p])
                
                # Correction for k=i:
                # k=i: a_i[i] - a_j[i] times b_v[p[i]] - b_u[p[i]]
                # p[i] = u
                # b_v[u] - b_u[u]
                # (a_ii - a_ji) * (b_vu - b_uu)
                
                # Correction for k=j:
                # k=j: a_i[j] - a_j[j] times b_v[p[j]] - b_u[p[j]]
                # p[j] = v
                # b_v[v] - b_u[v]
                # (a_ij - a_jj) * (b_vv - b_uv)
                
                # Let's compute the full dot product first
                # To avoid allocating large arrays in the inner loop, we can do:
                # np.dot(a_i, b_v[p]) - np.dot(a_i, b_u[p]) - np.dot(a_j, b_v[p]) + np.dot(a_j, b_u[p])
                # This is 4 dot products of length n.
                
                # dot1 = np.dot(a_i, b_v[p])
                # dot2 = np.dot(a_i, b_u[p])
                # dot3 = np.dot(a_j, b_v[p])
                # dot4 = np.dot(a_j, b_u[p])
                
                # sum_all = dot1 - dot2 - dot3 + dot4
                
                # term_i = (a_ii - a_ji) * (b_vu - b_uu)
                # term_j = (a_ij - a_jj) * (b_vv - b_uv)
                
                # delta = sum_all - term_i - term_j
                
                # Let's try to optimize this.
                # b_v_p = b_v[p] is a gather.
                # We can precompute b_p = b[p, :]? No, b is fixed, p changes.
                # b[p, :] is the matrix B' where B'[i,k] = b[p[i], k].
                # Then b_v_p = B'[v, :].
                # b_u_p = B'[u, :].
                
                # Let's maintain B_perm = b[np.ix_(p, p)]? No, that's for cost.
                # For delta, we need b[v, p[k]] which is b[v, p[k]].
                # If we have B_mat = b, then b[v, p[k]] is B_mat[v, p[k]].
                
                # Let's just use the 4 dot products. It's O(n) and NumPy is fast.
                
                # Optimization: Precompute b_v_p and b_u_p?
                # b_v_p = b[v, p]
                # b_u_p = b[u, p]
                
                # Actually, let's just calculate the delta directly.
                
                # delta = np.dot(a_i - a_j, b[v, p] - b[u, p])
                # This creates 3 temporary arrays of size n.
                # For n=150, this is fine.
                
                diff_a = a_i - a_j
                diff_b = b[v, p] - b[u, p]
                sum_all = np.dot(diff_a, diff_b)
                
                # Corrections
                # k = i
                # p[i] = u
                # term_i = (a_i[i] - a_j[i]) * (b[v, u] - b[u, u])
                # k = j
                # p[j] = v
                # term_j = (a_i[j] - a_j[j]) * (b[v, v] - b[u, v])
                
                term_i = (a_i[i] - a_j[i]) * (b[v, u] - b[u, u])
                term_j = (a_i[j] - a_j[j]) * (b[v, v] - b[u, v])
                
                delta = sum_all - term_i - term_j
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            # No improvement found
            break
        
        # Apply swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
    return p.tolist()
# EVOLVE-BLOCK-END
