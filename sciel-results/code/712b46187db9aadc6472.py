import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization using Hungarian Algorithm on linearized cost
    # W[i, j] represents the contribution of placing facility i at location j
    # Linearization: sum_j a[i,j] * b[p[i], p[j]] is not linear.
    # Standard proxy: W = A @ B.T
    W = a @ b.T
    row_ind, col_ind = linear_sum_assignment(W)
    p = np.full(n, -1, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Precompute some useful arrays for fast delta calculation
    # We need to be able to quickly compute:
    # sum_j a[i,j] * b[v, p[j]]
    # and sum_j a[j,i] * b[p[j], v]
    
    # Let's precompute B_perm = b[:, p] and B_perm_T = b[p, :]
    # But p changes.
    
    # Fast delta swap calculation without allocations
    # delta = sum_{j != i, k} [ (a[i,j]-a[k,j]) * (b[v, p[j]]-b[u, p[j]]) + (a[j,i]-a[j,k]) * (b[p[j],v]-b[p[j],u]) ]
    #        + diag/cross terms
    
    # We can precompute row sums of A and B? No, it depends on p.
    
    # Let's stick to the vectorized approach but optimize it.
    # Instead of computing diff_row and diff_col as full arrays, 
    # we can use the fact that:
    # sum_j a[i,j] * (b[v, p[j]] - b[u, p[j]]) = sum_j a[i,j]*b[v, p[j]] - sum_j a[i,j]*b[u, p[j]]
    
    # Let S1[i, v] = sum_j a[i,j] * b[v, p[j]]
    # This is (A @ B[:, p])[i, v]
    
    # Let's maintain:
    # A_Bp = a @ b[:, p]  (n x n)
    # Bp_A = b[p, :] @ a  (n x n) -> b[p, j] * a[j, i] = (b[p, :] @ a)[j, i]
    
    # Actually, let's just precompute the rows of b indexed by p.
    # B_perm = b[:, p]
    # Then A @ B_perm gives [i, v] = sum_j a[i, j] * b[v, p[j]]
    
    # Let's update these matrices when we swap.
    # If we swap p[i] and p[k]:
    # The columns of B_perm at positions i and k swap? No, B_perm is b[:, p].
    # B_perm[:, i] = b[:, p[i]]. If p[i] changes, the column changes.
    
    # Let's maintain:
    # S1 = a @ b[:, p]  (n x n) where S1[i, v] = sum_j a[i, j] b[v, p[j]]
    # S2 = b[p, :] @ a  (n x n) where S2[v, i] = sum_j b[p[j], v] a[j, i] = sum_j b[p[j], v] a[j, i]
    # Note: S2 is transposed relative to the second term in delta.
    # Term 2 in delta: sum_j (a[j, i] - a[j, k]) * (b[p[j], v] - b[p[j], u])
    # = sum_j a[j, i] b[p[j], v] - sum_j a[j, k] b[p[j], v] - sum_j a[j, i] b[p[j], u] + sum_j a[j, k] b[p[j], u]
    # = (S2.T)[v, i] - (S2.T)[v, k] - (S2.T)[u, i] + (S2.T)[u, k]
    # Let's define S2 such that S2[i, v] = sum_j a[j, i] b[p[j], v].
    # Then S2 = a.T @ b[p, :].
    
    # Let's initialize S1 and S2.
    B_perm = b[:, p]
    S1 = a @ B_perm  # S1[i, v] = sum_j a[i, j] b[v, p[j]]
    
    # S2[i, v] = sum_j a[j, i] b[p[j], v]
    # S2 = a.T @ b[p, :]
    B_perm_T = b[p, :]
    S2 = a.T @ B_perm_T # S2[i, v] = sum_j a[j, i] b[p[j], v]
    
    def get_cost(p):
        # Can be computed from S1 and S2?
        # Cost = sum_{i, j} a[i, j] b[p[i], p[j]]
        # = sum_i sum_j a[i, j] b[p[i], p[j]]
        # This is not directly S1 or S2.
        # S1[i, p[i]] = sum_j a[i, j] b[p[i], p[j]]. Sum over i gives cost.
        return int(np.sum(S1[np.arange(n), p]))

    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for fast delta calculation
    # We need to update S1 and S2 when swapping p[i] and p[k].
    # Let u = p[i], v = p[k].
    # New p: p[i]=v, p[k]=u.
    # B_perm changes: column i becomes b[:, v], column k becomes b[:, u].
    # S1 = a @ B_perm.
    # S1_new[:, i] = a @ b[:, v]
    # S1_new[:, k] = a @ b[:, u]
    # Other columns unchanged.
    
    # S2 = a.T @ B_perm_T.
    # B_perm_T changes: row i becomes b[v, :], row k becomes b[u, :].
    # S2_new[i, :] = b[v, :] @ a
    # S2_new[k, :] = b[u, :] @ a
    # Other rows unchanged.
    
    # To make this fast, we can precompute:
    # A_b_cols = a @ b  (n x n) where (A_b_cols)[:, v] = a @ b[:, v]
    # b_rows_a = b @ a  (n x n) where (b_rows_a)[v, :] = b[v, :] @ a
    
    A_b = a @ b
    b_a = b @ a
    
    # Now, to compute delta for swap i, k:
    # u = p[i], v = p[k]
    
    # Term 1: sum_j (a[i,j]-a[k,j]) * (b[v, p[j]]-b[u, p[j]])
    # = (sum_j a[i,j]b[v,p[j]] - sum_j a[i,j]b[u,p[j]]) - (sum_j a[k,j]b[v,p[j]] - sum_j a[k,j]b[u,p[j]])
    # = (S1[i, v] - S1[i, u]) - (S1[k, v] - S1[k, u])
    
    # Term 2: sum_j (a[j,i]-a[j,k]) * (b[p[j],v]-b[p[j],u])
    # = (sum_j a[j,i]b[p[j],v] - sum_j a[j,i]b[p[j],u]) - (sum_j a[j,k]b[p[j],v] - sum_j a[j,k]b[p[j],u])
    # = (S2[i, v] - S2[i, u]) - (S2[k, v] - S2[k, u])
    
    # Total "bulk" delta = Term 1 + Term 2
    
    # But this sum includes j=i and j=k.
    # For j=i:
    # Term 1 part: (a[i,i]-a[k,i]) * (b[v, p[i]] - b[u, p[i]]) = (a[i,i]-a[k,i]) * (b[v, u] - b[u, u])
    # Term 2 part: (a[i,i]-a[i,k]) * (b[p[i], v] - b[p[i], u]) = (a[i,i]-a[i,k]) * (b[u, v] - b[u, u])
    
    # For j=k:
    # Term 1 part: (a[i,k]-a[k,k]) * (b[v, p[k]] - b[u, p[k]]) = (a[i,k]-a[k,k]) * (b[v, v] - b[u, v])
    # Term 2 part: (a[k,i]-a[k,k]) * (b[p[k], v] - b[p[k], u]) = (a[k,i]-a[k,k]) * (b[v, v] - b[v, u])
    
    # So bulk_delta = (S1[i, v] - S1[i, u] - S1[k, v] + S1[k, u]) + (S2[i, v] - S2[i, u] - S2[k, v] + S2[k, u])
    
    # Correction for j=i, k:
    # corr_i = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
    # corr_k = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
    
    # The "off-diag" part (j != i, k) is bulk_delta - corr_i - corr_k
    
    # The new diagonal/cross terms for i, k in the new configuration:
    # New p[i]=v, p[k]=u.
    # Terms involving i, k in new cost:
    # a[i,i]b[v,v] + a[i,k]b[v,u] + a[k,i]b[u,v] + a[k,k]b[u,u]
    # Old terms:
    # a[i,i]b[u,u] + a[i,k]b[u,v] + a[k,i]b[v,u] + a[k,k]b[v,v]
    
    # New - Old for these 4 terms:
    # diag_new = a[i,i]*(b[v,v]-b[u,u]) + a[k,k]*(b[u,u]-b[v,v]) + a[i,k]*(b[v,u]-b[u,v]) + a[k,i]*(b[u,v]-b[v,u])
    
    # Total delta = (bulk_delta - corr_i - corr_k) + diag_new
    
    def compute_delta_swap(i, k):
        u, v = p[i], p[k]
        if u == v: return 0
        
        # Bulk terms
        bulk = (S1[i, v] - S1[i, u] - S1[k, v] + S1[k, u]) + \
               (S2[i, v] - S2[i, u] - S2[k, v] + S2[k, u])
        
        # Corrections
        corr_i = (a[i,i]-a[k,i])*(b[v,u]-b[u,u]) + (a[i,i]-a[i,k])*(b[u,v]-b[u,u])
        corr_k = (a[i,k]-a[k,k])*(b[v,v]-b[u,v]) + (a[k,i]-a[k,k])*(b[v,v]-b[v,u])
        
        # New diagonal/cross
        diag_new = a[i,i]*(b[v,v]-b[u,u]) + a[k,k]*(b[u,u]-b[v,v]) + \
                   a[i,k]*(b[v,u]-b[u,v]) + a[k,i]*(b[u,v]-b[v,u])
        
        return bulk - corr_i - corr_k + diag_new

    def apply_swap(i, k):
        nonlocal S1, S2, p, current_cost
        u, v = p[i], p[k]
        if u == v: return
        
        # Update S1
        # S1[:, i] = A_b[:, v]
        # S1[:, k] = A_b[:, u]
        S1[:, i] = A_b[:, v]
        S1[:, k] = A_b[:, u]
        
        # Update S2
        # S2[i, :] = b_a[v, :]
        # S2[k, :] = b_a[u, :]
        S2[i, :] = b_a[v, :]
        S2[k, :] = b_a[u, :]
        
        # Update p
        p[i], p[k] = v, u
        
        # Update cost
        current_cost += compute_delta_swap(i, k) # Wait, compute_delta uses old p?
        # No, compute_delta_swap uses current p. We should compute delta BEFORE updating p.
        
        # Let's restructure.
        
    # Corrected apply_swap
    def apply_swap(i, k):
        nonlocal S1, S2, p, current_cost
        u, v = p[i], p[k]
        if u == v: return 0
        
        delta = compute_delta_swap(i, k)
        
        # Update S1
        S1[:, i] = A_b[:, v]
        S1[:, k] = A_b[:, u]
        
        # Update S2
        S2[i, :] = b_a[v, :]
        S2[k, :] = b_a[u, :]
        
        # Update p
        p[i], p[k] = v, u
        
        # Update cost
        current_cost += delta
        return delta

    # 2. Simulated Annealing
    # Estimate T
    sample_deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        d = compute_delta_swap(i, k)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 2.0 if sample_deltas else 10000.0
    T_min = 1.0
    alpha = 0.9998
    
    start_time = time.time()
    time_limit = 60.0
    
    # Use random integers for speed
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        k = np.random.randint(n)
        if i == k: continue
        
        delta = compute_delta_swap(i, k)
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            apply_swap(i, k)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    # 3. Fast Local Search (2-opt Descent)
    # To speed up, we can do a few passes of random sampling of pairs if full O(n^2) is too slow.
    # But n=150, O(n^2) = 11250 pairs. 10 passes = 112k deltas.
    # Each delta is ~10 float ops. 1M ops is fast.
    # Let's try full scan.
    
    p = best_p.copy()
    current_cost = best_cost
    # Reinitialize S1, S2 for this p
    B_perm = b[:, p]
    S1 = a @ B_perm
    B_perm_T = b[p, :]
    S2 = a.T @ B_perm_T
    
    improved = True
    max_passes = 10
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            for k in range(i+1, n):
                delta = compute_delta_swap(i, k)
                if delta < 0:
                    apply_swap(i, k)
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
    # 4. Second SA Round
    p = best_p.copy()
    current_cost = best_cost
    # Reinitialize S1, S2
    B_perm = b[:, p]
    S1 = a @ B_perm
    B_perm_T = b[p, :]
    S2 = a.T @ B_perm_T
    
    # Re-estimate T
    sample_deltas = []
    for _ in range(100):
        i, k = np.random.choice(n, 2, replace=False)
        d = compute_delta_swap(i, k)
        sample_deltas.append(abs(d))
    
    T = np.mean(sample_deltas) * 1.5 if sample_deltas else 5000.0
    T_min = 1.0
    alpha = 0.9999
    
    start_time = time.time()
    time_limit = 20.0
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        k = np.random.randint(n)
        if i == k: continue
        
        delta = compute_delta_swap(i, k)
        
        if delta < 0 or (T > T_min and np.random.rand() < np.exp(-delta / T)):
            apply_swap(i, k)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min

    # 5. Final Local Search
    p = best_p.copy()
    current_cost = best_cost
    # Reinitialize S1, S2
    B_perm = b[:, p]
    S1 = a @ B_perm
    B_perm_T = b[p, :]
    S2 = a.T @ B_perm_T
    
    improved = True
    max_passes = 5
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            for k in range(i+1, n):
                delta = compute_delta_swap(i, k)
                if delta < 0:
                    apply_swap(i, k)
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()

    return best_p.tolist()
