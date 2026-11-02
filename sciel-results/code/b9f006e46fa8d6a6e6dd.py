# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = 4, 4, 8
    
    # 1. Construct Schoolbook Decomposition (Rank 128)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Convert to list format as required by interface
    def to_list(mat):
        return mat.tolist()
        
    best_U = to_list(U_school)
    best_V = to_list(V_school)
    best_W = to_list(W_school)
    best_rank = rank_school
    
    # 2. Attempt to find lower rank via vectorized ALS
    # We try a few ranks below schoolbook.
    # Note: Exact integer/half-integer solutions are rare in random continuous search.
    # We will check for near-exact solutions and round them.
    
    def try_rank(rank):
        if rank >= best_rank:
            return None
        if time.time() > deadline - 5:
            return None
            
        # Try a few random restarts
        for seed in range(5):
            if time.time() > deadline - 5:
                return None
            rng = np.random.RandomState(seed + rank)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Target tensor T[i,j,k] = 1
            # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Vectorized ALS
            # Precompute indices or use broadcasting
            # Shape of T: (n, m, p)
            T_target = np.ones((n, m, p))
            
            for it in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                # For each (i,j), we solve for U[i*m+j, :]
                # Eq: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let M_U[i,j] be a p x rank matrix. M_U[i,j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                # We can build M_U for all i,j using broadcasting?
                # V is (m*p, rank). W is (p*n, rank).
                # Reshape V to (m, p, rank). Reshape W to (p, n, rank).
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                # M_U[i, j, k, r] = V_reshaped[j, k, r] * W_reshaped[k, i, r]
                # We need to sum over k for each (i,j) to get the linear system?
                # No, for fixed (i,j), the variable is U[i*m+j, :] (length rank).
                # The equation is: sum_r U[i*m+j, r] * (sum_k? No, k is the output dimension of the system)
                # Wait. The tensor entry is T[i,j,k].
                # For fixed i,j, we have p equations (one for each k) and rank unknowns (U[i*m+j, 0..rank-1]).
                # Equation k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is a linear system: A_{i,j} @ u_{i,j} = b_{i,j}
                # Where A_{i,j} is p x rank. A_{i,j}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # b_{i,j} is p x 1, all ones.
                
                # Construct A for all i, j
                # A[i, j, k, r] = V[j, k, r] * W[k, i, r]
                # V_reshaped: (m, p, rank) -> (1, m, p, rank)
                # W_reshaped: (p, n, rank) -> (p, 1, 1, rank)
                # We need to align indices.
                # A[i, j, k, r] depends on j, k, r from V and k, i, r from W.
                # Let's iterate over i, j to build the system, it's small (4x4).
                
                for i in range(n):
                    for j in range(m):
                        # A[k, r] for k in 0..p-1, r in 0..rank-1
                        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Vectorized:
                        V_slice = V[j*p : j*p+p, :] # (p, rank)
                        W_slice = W[0::n, i]        # (p, rank) -- Wait, W index is k*n+i
                        # W is (p*n, rank). Indices: k*n+i.
                        # So for fixed i, the rows are i, n+i, 2n+i, ...
                        W_slice = W[i::n, :] # (p, rank)
                        
                        A = V_slice * W_slice # (p, rank)
                        b = np.ones(p)
                        
                        # Solve A @ u = b
                        # Use least squares
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m + j, :] = u
                        except:
                            pass
                
                # Update V
                # For fixed (j, k), variable is V[j*p+k, :] (length rank).
                # Equation i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # System: B_{j,k} @ v_{j,k} = c_{j,k}
                # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # c[i] = 1
                
                for j in range(m):
                    for k in range(p):
                        # B[i, r] for i in 0..n-1
                        # U indices: i*m+j. Rows j, m+j, 2m+j...
                        U_slice = U[j::m, :] # (n, rank)
                        # W indices: k*n+i. Rows k*n, k*n+1...
                        W_slice = W[k*n : k*n+n, :] # (n, rank)
                        
                        B = U_slice * W_slice # (n, rank)
                        c = np.ones(n)
                        
                        try:
                            v, _, _, _ = np.linalg.lstsq(B, c, rcond=None)
                            V[j*p + k, :] = v
                        except:
                            pass
                
                # Update W
                # For fixed (k, i), variable is W[k*n+i, :] (length rank).
                # Equation j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # System: C_{k,i} @ w_{k,i} = d_{k,i}
                # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # d[j] = 1
                
                for k in range(p):
                    for i in range(n):
                        # C[j, r] for j in 0..m-1
                        # U indices: i*m+j. Rows i*m, i*m+1...
                        U_slice = U[i*m : i*m+m, :] # (m, rank)
                        # V indices: j*p+k. Rows k, p+k, 2p+k...
                        V_slice = V[k::p, :] # (m, rank)
                        
                        C = U_slice * V_slice # (m, rank)
                        d = np.ones(m)
                        
                        try:
                            w, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
                            W[k*n + i, :] = w
                        except:
                            pass
                
                # Check error
                # Calculate T_reconstructed
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Vectorized check:
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # U: (n, m, rank)
                # V: (m, p, rank)
                # W: (p, n, rank)
                
                # Product term for each r:
                # P_r[i, j, k] = U[i, j, r] * V[j, k, r] * W[k, i, r]
                # Sum over r.
                
                # This is a bit tricky to vectorize directly without expanding rank.
                # Rank is small (<=128). n,m,p are small.
                # We can just loop over r or use einsum.
                # np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
                # Note spaces in einsum string must be removed for numpy < 1.20? No, standard is 'ijr,jkr,ki r->ijk' is invalid, use 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r->ijk' is wrong syntax.
                # Correct: 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r->ijk'
                # Actually: U[i,j,r], V[j,k,r], W[k,i,r].
                # np.einsum('ijr,jkr,ki r->ijk', ...) is wrong.
                # np.einsum('ijr,jkr,ki r->ijk', U, V, W)
                # Let's use: np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
                # Wait, the string is 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r->ijk'
                # The space between 'ki' and 'r' is not allowed.
                # It should be 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r->ijk'
                # Correct string: 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r->ijk'
                # Actually: 'ijr,jkr,ki r->ijk' is wrong.
                # Indices:
                # U: i, j, r
                # V: j, k, r
                # W: k, i, r
                # Sum over r.
                # Output: i, j, k
                # String: 'ijr,jkr,ki r->ijk' -> 'ijr,jkr,ki r->ijk'
                # Let's write it carefully: 'ijr,jkr,ki r->ijk'
                # The W indices are k, i, r. So 'ki r' is 'kir'.
                # String: 'ijr,jkr,kir->ijk'
                
                try:
                    T_rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                    error = np.sum((T_rec - 1.0)**2)
                except:
                    error = float('inf')
                
                if error < 1e-5:
                    # Found a good candidate. Try to round to half-integers.
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded version
                    U_r_reshaped = U_rounded.reshape(n, m, rank)
                    V_r_reshaped = V_rounded.reshape(m, p, rank)
                    W_r_reshaped = W_rounded.reshape(p, n, rank)
                    
                    try:
                        T_rec_r = np.einsum('ijr,jkr,kir->ijk', U_r_reshaped, V_r_reshaped, W_r_reshaped)
                        error_r = np.sum((T_rec_r - 1.0)**2)
                    except:
                        error_r = float('inf')
                    
                    if error_r < 1e-4:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            return None
            
    # Try a few lower ranks
    # Schoolbook is 128.
    # We try 120, 118, 116...
    # Given the difficulty, we might not find any.
    for r in range(126, 100, -2):
        if time.time() > deadline - 10:
            break
        res = try_rank(r)
        if res:
            best_U = res["U"]
            best_V = res["V"]
            best_W = res["W"]
            best_rank = r
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
