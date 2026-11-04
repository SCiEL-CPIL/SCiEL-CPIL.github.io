"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    # Ensure float64 for precision and speed
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # Precompute transposed b for column access speed if needed, 
    # but b[p, k] is row access, b[k, p] is col access.
    
    # 1. Initialization: Rank Matching
    # Heavy flows (high row sum in A) should go to short distances (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow
    heavy_idx = np.argsort(-a_sums, kind='stable')
    # Sort locations by ascending distance
    central_idx = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Compute initial cost
    # Cost = sum(a[i, j] * b[p[i], p[j]])
    # Vectorized: np.einsum('ij,ik,jk->', a, b[p][:, p]) is not quite right.
    # b[p][:, p] creates a matrix where row i is b[p[i], :] and we take col p[i]?
    # No, b[p] reorders rows. b[p][:, p] takes row p[i] and col p[i]? 
    # Actually, np.einsum('ij,kl->', a, b[p][:, p]) works if b[p] is the permuted matrix.
    # Let's just use a simple loop or einsum for initial cost to be safe and clear.
    # For n=150, a loop is fine, but einsum is faster.
    # b_perm = b[p][:, p] -> Matrix where element (i,j) is b[p[i], p[j]]
    # Cost = np.sum(a * b[p][:, p])
    
    def compute_cost(p):
        return np.sum(a * b[p][:, p])

    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    T = 1000000.0  # Initial temperature, scale based on cost magnitude
    # Estimate typical delta magnitude
    # Cost is around 8e6. Deltas might be in range +/- 1e4 or 1e5.
    # Let's set T based on average absolute value of a and b elements?
    # Or just start high and cool.
    T = max(1.0, current_cost / 1000.0)
    cooling_rate = 0.9995
    min_T = 1.0
    
    # Precompute some things for speed?
    # A and B are fixed. P changes.
    
    while time.time() < deadline:
        # Pick random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Calculate Delta for swapping p[i] and p[j]
        # Delta = sum_{k != i,j} (A[i,k]-A[j,k])*(B[v,p[k]]-B[u,p[k]]) + 
        #         (A[k,i]-A[k,j])*(B[p[k],v]-B[p[k],u]) +
        #         A[i,j]*(B[v,u]-B[u,v]) + A[j,i]*(B[u,v]-B[v,u])
        
        # Vectorized calculation
        # Create mask to exclude i and j
        # mask = np.ones(n, dtype=bool); mask[i]=0; mask[j]=0
        
        # To avoid creating mask every time (though n=150 is small), we can compute full sums and subtract terms for i and j.
        
        # Term 1: sum_k A[i,k] (B[v,p[k]] - B[u,p[k]])
        # B[v, p] is vector of length n. B[u, p] is vector of length n.
        diff_b_vu = b[v, p] - b[u, p]
        sum1 = np.dot(a[i, :], diff_b_vu)
        
        # Term 2: sum_k A[k,i] (B[p[k],v] - B[p[k],u])
        # B[p, v] is vector of length n (column v of b indexed by p)
        # B[p, u] is vector of length n
        diff_b_pv_u = b[p, v] - b[p, u]
        sum2 = np.dot(a[:, i], diff_b_pv_u)
        
        # Term 3: sum_k A[j,k] (B[u,p[k]] - B[v,p[k]])
        # This is - sum1 if we swap u,v and i,j? No.
        # sum3 = np.dot(a[j, :], b[u, p] - b[v, p])
        sum3 = np.dot(a[j, :], -diff_b_vu) # b[u,p]-b[v,p] = -(b[v,p]-b[u,p])
        
        # Term 4: sum_k A[k,j] (B[p[k],u] - B[p[k],v])
        sum4 = np.dot(a[:, j], -diff_b_pv_u)
        
        # Sum so far includes k=i and k=j terms which need correction.
        # Let's identify the terms where k=i or k=j in the sums above.
        
        # In sum1 (k=i): A[i,i] * (B[v, p[i]] - B[u, p[i]]) = A[i,i] * (B[v, u] - B[u, u])
        # In sum1 (k=j): A[i,j] * (B[v, p[j]] - B[u, p[j]]) = A[i,j] * (B[v, v] - B[u, v])
        
        # In sum2 (k=i): A[i,i] * (B[p[i], v] - B[p[i], u]) = A[i,i] * (B[u, v] - B[u, u])
        # In sum2 (k=j): A[j,i] * (B[p[j], v] - B[p[j], u]) = A[j,i] * (B[v, v] - B[v, u])
        
        # In sum3 (k=i): A[j,i] * (B[u, u] - B[v, u])
        # In sum3 (k=j): A[j,j] * (B[u, v] - B[v, v])
        
        # In sum4 (k=i): A[i,j] * (B[u, u] - B[v, u])
        # In sum4 (k=j): A[j,j] * (B[u, v] - B[v, v])
        
        # This manual subtraction is error-prone. 
        # Given n=150, it is safer and fast enough to use a mask.
        
        mask = np.ones(n, dtype=bool)
        mask[i] = 0
        mask[j] = 0
        
        # Recalculate with mask
        # sum1_masked = np.dot(a[i, mask], diff_b_vu[mask])
        # sum2_masked = np.dot(a[mask, i], diff_b_pv_u[mask])
        # sum3_masked = np.dot(a[j, mask], -diff_b_vu[mask])
        # sum4_masked = np.dot(a[mask, j], -diff_b_pv_u[mask])
        
        # Delta = sum1_masked + sum2_masked + sum3_masked + sum4_masked + 
        #         A[i,j]*(B[v,u]-B[u,v]) + A[j,i]*(B[u,v]-B[v,u])
        
        # Optimization: Pre-allocate mask? No, i,j change.
        # n=150, boolean indexing is fast.
        
        s1 = np.dot(a[i, mask], diff_b_vu[mask])
        s2 = np.dot(a[mask, i], diff_b_pv_u[mask])
        s3 = np.dot(a[j, mask], -diff_b_vu[mask])
        s4 = np.dot(a[mask, j], -diff_b_pv_u[mask])
        
        direct = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        delta = s1 + s2 + s3 + s4 + direct
        
        if delta < 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # SA Acceptance
            if T > min_T:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
            # Cooling
            T *= cooling_rate
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
