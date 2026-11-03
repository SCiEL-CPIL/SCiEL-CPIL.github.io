import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sum in A) to central locations (low row sum in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow weight (descending)
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    # Sort locations by centrality (ascending - lower sum is more central)
    central_locations = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    # Compute initial cost and S matrix
    # S[i, u] = sum_k a[i, k] * b[u, p[k]]
    # B_perm[u, k] = b[u, p[k]]
    B_perm = b[:, p]
    S = a @ B_perm
    
    # Helper to compute cost
    def get_cost(p_curr):
        # sum_{i,j} a[i,j] b[p[i], p[j]]
        # Can be computed as trace(A * B_permuted) or sum
        # Using S is not direct for total cost, but we can maintain it.
        # Or just compute directly for check.
        # Direct computation:
        b_perm = b[:, p_curr]
        # Cost = sum_{i,j} a[i,j] b_perm[i,j]? No.
        # b_perm[i,j] = b[i, p[j]].
        # We want sum_{i,j} a[i,j] b[p[i], p[j]].
        # Let B_p be matrix where B_p[i,j] = b[p[i], p[j]].
        # Cost = np.sum(a * B_p).
        # B_p can be constructed as b[np.ix_(p, p)].
        B_p = b[np.ix_(p_curr, p_curr)]
        return np.sum(a * B_p)
    
    best_cost = get_cost(p)
    best_p = p.copy()
    
    current_cost = best_cost
    current_p = p.copy()
    
    # Local Search Loop
    while time.time() < deadline:
        # Find best swap (i, j)
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Iterate over all pairs
        # We can vectorize the delta calculation?
        # Delta(i, j) = 2 * (S[i, P(j)] + S[j, P(i)] - S[i, P(i)] - S[j, P(j)] + b[P(i), P(j)] * (2*a[i,j] - a[i,i] - a[j,j]))
        
        # Precompute parts that don't depend on j for a fixed i?
        # Or just loop i and vectorize over j?
        
        # Let's try vectorizing over j for each i.
        # P_i = current_p[i]
        # Term 1: S[i, P(j)] -> S[i, current_p] (vector of length n)
        # Term 2: S[j, P(i)] -> S[:, P_i] (vector of length n, indexed by j)
        # Term 3: S[i, P(i)] -> scalar
        # Term 4: S[j, P(j)] -> S[j, current_p[j]] (diagonal of S with respect to p)
        # Term 5: b[P(i), P(j)] * (2*a[i,j] - a[i,i] - a[j,j])
        
        # Let's compute diagonal term D_j = S[j, current_p[j]]
        diag_S = S[np.arange(n), current_p]
        
        # For each i:
        for i in range(n):
            Pi = current_p[i]
            Sj_Pi = S[:, Pi] # Vector S[j, Pi] for all j
            
            # S[i, P(j)] is S[i, current_p]
            Si_Pj = S[i, current_p]
            
            # Term 3 is constant for this i
            term3 = S[i, Pi]
            
            # Term 4 is diag_S
            term4 = diag_S
            
            # Term 5: b[Pi, current_p] * (2*a[i, :] - a[i,i] - a[:, i])
            # b[Pi, current_p] is vector b[Pi, current_p[j]]
            b_Pi_Pj = b[Pi, current_p]
            coeff = 2 * a[i, :] - a[i, i] - a[:, i]
            term5 = b_Pi_Pj * coeff
            
            # Total Delta / 2
            deltas = Si_Pj + Sj_Pi - term3 - term4 + term5
            
            # Mask out j == i
            deltas[i] = np.inf
            
            # Find min
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_i = i
                best_j = min_idx
        
        if best_i == -1 or best_delta >= 0:
            # No improvement found. Perturb to escape local optimum.
            # Randomly swap 2 or 3 pairs
            if time.time() > deadline - 10:
                break
            
            # Perturb
            k = 2
            indices = np.random.choice(n, k, replace=False)
            for idx in indices:
                # Swap with a random other index
                j = np.random.randint(n)
                if j == idx:
                    j = (j + 1) % n
                current_p[[idx, j]] = current_p[[j, idx]]
            
            # Update S and cost
            B_perm = b[:, current_p]
            S = a @ B_perm
            current_cost = get_cost(current_p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
            continue
        
        # Apply swap
        i, j = best_i, best_j
        Pi, Pj = current_p[i], current_p[j]
        
        # Update current_p
        current_p[[i, j]] = [Pj, Pi]
        
        # Update S
        # S_new[k, u] = S_old[k, u] + (A[k, i] - A[k, j]) * (B_old[u, j] - B_old[u, i])
        # B_old[u, j] = b[u, Pj]
        # B_old[u, i] = b[u, Pi]
        # Vector D_u = b[:, Pj] - b[:, Pi]
        D = b[:, Pj] - b[:, Pi]
        # Vector V_k = A[k, i] - A[k, j]
        V = a[:, i] - a[:, j]
        
        S += np.outer(V, D)
        
        # Update cost
        current_cost += 2 * best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
            
        # Safety check
        if time.time() > deadline:
            break
            
    return best_p.tolist()
