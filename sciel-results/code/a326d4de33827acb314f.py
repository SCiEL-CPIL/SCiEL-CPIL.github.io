import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition via random search
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = _search_low_rank(target_rank, remaining - 5)
        if result is not None:
            return result
    
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using random restarts."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # The tensor T[(i,j),(j',k),(i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    # Non-zero entries: when a=(i,j), b=(j,k), c=(i,k) for i in {0,1}, j in {0,1}, k in {0..7}
    # That's 2*2*8 = 32 non-zero entries, each equal to 1.
    # All other entries must be 0.
    
    # Build the list of (a,b,c) indices where T=1
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j  # A index: row i, col j
                b = j * 8 + k  # B index: row j, col k
                c = k * 2 + i  # C index: col k, row i (convention W[k*n+i])
                nonzero_indices.append((a, b, c))
    
    # Total number of (a,b,c) triples
    total_triples = num_a * num_b * num_c  # 4*16*16 = 1024
    
    # For each random restart, initialize U,V,W randomly and try to minimize residual
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        # Random initialization with small integers/half-integers
        rng = np.random.default_rng(seed=restart)
        U = rng.integers(-2, 3, size=(num_a, target_rank)).astype(float)
        V = rng.integers(-2, 3, size=(num_b, target_rank)).astype(float)
        W = rng.integers(-2, 3, size=(num_c, target_rank)).astype(float)
        
        # Check residual on non-zero entries
        # For each (a,b,c) in nonzero_indices: sum_r U[a,r]*V[b,r]*W[c,r] should be 1
        # For other (a,b,c): sum should be 0
        
        # Efficient check: compute all products
        # Product[r] = U[:,r] * V[:,r] gives a (4,16) matrix for each r
        # Then for each c, sum_r W[c,r] * (U*V)[:,r]
        
        # Let's compute the full tensor and check
        # T_hat[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Use einsum for efficiency
        # First compute UV[a,b,r] = U[a,r]*V[b,r]
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (4, 16, rank)
        # Then T_hat[a,b,c] = sum_r UV[a,b,r] * W[c,r]
        T_hat = np.einsum('abr,cr->abc', UV, W)  # (4, 16, 16)
        
        # Check non-zero entries
        max_err = 0
        for a, b, c in nonzero_indices:
            err = abs(T_hat[a, b, c] - 1)
            max_err = max(max_err, err)
        
        # Check zero entries (sample a few for speed)
        if max_err < 1e-6:
            # Verify all zero entries
            all_zero_ok = True
            for a in range(num_a):
                for b in range(num_b):
                    for c in range(num_c):
                        if (a, b, c) not in nonzero_indices:
                            if abs(T_hat[a, b, c]) > 1e-6:
                                all_zero_ok = False
                                break
                    if not all_zero_ok:
                        break
                if not all_zero_ok:
                    break
            
            if all_zero_ok:
                # Round to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
