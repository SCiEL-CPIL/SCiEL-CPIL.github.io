import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s
    
    # 1. Initialization: Rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This can be computed as trace(a @ b[p][:, p])? No.
        # It's sum_i sum_j a[i,j] b[p_i, p_j]
        # Let B_p be b with rows and cols permuted by p.
        # Then cost is sum_{i,j} a[i,j] B_p[i,j] = np.sum(a * B_p)
        # But B_p is b[np.ix_(p, p)]
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for speed?
    # We will use the O(n) delta calculation.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # To avoid bias, we can shuffle the order of pairs or just iterate i < j
        # Iterating i < j is standard.
        for i in range(n - 1):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                k = p[i]
                l = p[j]
                
                # Calculate delta
                # Delta = (A[i,i]-A[j,j])*(B[l,l]-B[k,k])/2? No.
                # Let's use the derived formula:
                # Delta = A[i,i](B[l,l]-B[k,k]) + A[j,j](B[k,k]-B[l,l]) + A[i,j](B[l,k]-B[k,l]) + A[j,i](B[k,l]-B[l,k])
                # + sum_{u != i,j} ( (A[u,i]-A[u,j])*(B[p[u],l]-B[p[u],k]) + (A[i,u]-A[j,u])*(B[l,p[u]]-B[k,p[u]]) )
                
                # Vectorized sum part
                # u ranges from 0 to n-1
                # We compute for all u, then subtract u=i and u=j contributions if they were non-zero in the formula?
                # The formula for Term_u uses A[u,i], etc.
                # If u=i: Term_i = (A[i,i]-A[i,j])*(B[p[i],l]-B[p[i],k]) + (A[i,i]-A[j,i])*(B[l,p[i]]-B[k,p[i]])
                # p[i]=k, so B[k,l]-B[k,k] and B[l,k]-B[k,k].
                # This gets messy. It's safer to just mask out i and j in the vector operation.
                
                # Create mask
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                
                u_idx = np.where(mask)[0]
                
                # Term 1: (A[u,i]-A[u,j]) * (B[p[u],l]-B[p[u],k])
                # A[u,i] is column i of A. A[u,j] is column j of A.
                # B[p[u], l] is row p[u], col l of B.
                
                # This is still O(n) with Python overhead if not careful.
                # Let's try to vectorize the whole sum.
                
                # Col_i = a[:, i]
                # Col_j = a[:, j]
                # Row_i = a[i, :]
                # Row_j = a[j, :]
                
                # We need sum over u in u_idx of:
                # (a[u,i] - a[u,j]) * (b[p[u], l] - b[p[u], k]) + (a[i,u] - a[j,u]) * (b[l, p[u]] - b[k, p[u]])
                
                # Let's compute arrays for all u first, then sum the masked part.
                # This is O(n) memory and O(n) time.
                
                # diff_a_col = a[:, i] - a[:, j]
                # diff_a_row = a[i, :] - a[j, :]
                
                # b_diff_col = b[p, l] - b[p, k]  # vector of size n, index u -> b[p[u], l] - b[p[u], k]
                # b_diff_row = b[l, p] - b[k, p]  # vector of size n, index u -> b[l, p[u]] - b[k, p[u]]
                
                # sum_part = np.sum(diff_a_col * b_diff_col + diff_a_row * b_diff_row)
                # But we must exclude i and j from the sum.
                
                # Let's compute the full sum and subtract the i and j terms.
                # Full sum S = sum_{u=0}^{n-1} T_u
                # We want S' = S - T_i - T_j
                
                diff_a_col = a[:, i] - a[:, j]
                diff_a_row = a[i, :] - a[j, :]
                
                # b[p, l] is b at rows p, col l.
                b_pl = b[p, l]
                b_pk = b[p, k]
                b_l_p = b[l, p]
                b_k_p = b[k, p]
                
                b_diff_col = b_pl - b_pk
                b_diff_row = b_l_p - b_k_p
                
                T_all = diff_a_col * b_diff_col + diff_a_row * b_diff_row
                S = np.sum(T_all)
                
                # Calculate T_i
                # u = i
                # T_i = (a[i,i]-a[i,j])*(b[p[i],l]-b[p[i],k]) + (a[i,i]-a[j,i])*(b[l,p[i]]-b[k,p[i]])
                # p[i] = k
                T_i = (a[i,i]-a[i,j])*(b[k,l]-b[k,k]) + (a[i,i]-a[j,i])*(b[l,k]-b[k,k])
                
                # Calculate T_j
                # u = j
                # T_j = (a[j,i]-a[j,j])*(b[p[j],l]-b[p[j],k]) + (a[i,j]-a[j,j])*(b[l,p[j]]-b[k,p[j]])
                # p[j] = l
                T_j = (a[j,i]-a[j,j])*(b[l,l]-b[l,k]) + (a[i,j]-a[j,j])*(b[l,l]-b[k,l])
                
                sum_part = S - T_i - T_j
                
                # Constant part
                const_part = (a[i,i] - a[j,j]) * (b[l,l] - b[k,k]) + (a[i,j] - a[j,i]) * (b[l,k] - b[k,l])
                # Wait, my previous derivation:
                # A[i,i](B[l,l]-B[k,k]) + A[j,j](B[k,k]-B[l,l]) = (A[i,i]-A[j,j])(B[l,l]-B[k,k])
                # A[i,j](B[l,k]-B[k,l]) + A[j,i](B[k,l]-B[l,k]) = (A[i,j]-A[j,i])(B[l,k]-B[k,l])
                
                delta = const_part + sum_part
                
                if delta < -1e-9: # Improvement
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
    return best_p.tolist()
