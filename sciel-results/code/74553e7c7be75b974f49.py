    def construct():
        n, m, p = 4, 4, 8
        
        # Build 4x4x4 Strassen decomposition (rank 49)
        U_sub, V_sub, W_sub = _build_recursive_strassen(4)
        rank_sub = U_sub.shape[1] # 49
        
        # Total rank for 4x4x8 is 2 * 49 = 98
        rank = 2 * rank_sub
        
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Part 1: Compute C[:, 0:4] = A @ B[:, 0:4]
        # U is same
        U[:, 0:rank_sub] = U_sub
        
        # V_sub is 4x4 (flattened 16). Global V is 4x8 (flattened 32).
        # V_sub index: k*4 + j (k in 0..3, j in 0..3)
        # V_global index for block 1: k*8 + j
        for k in range(4):
            for j in range(4):
                V[k*8 + j, 0:rank_sub] = V_sub[k*4 + j, 0:rank_sub]
        
        # W_sub is 4x4 (flattened 16). Global W is 8x4 (flattened 32).
        # W_sub index: k*4 + i (k in 0..3, i in 0..3) -- Note: W is (p,n) so row is p-index?
        # In _build_recursive_strassen, W is (n*n, rank) for n=n.
        # Index used: k_global*n + i_global.
        # So W_sub index is k*4 + i.
        # Global W index for block 1 (C cols 0-3): k*4 + i.
        for k in range(4):
            for i in range(4):
                W[k*4 + i, 0:rank_sub] = W_sub[k*4 + i, 0:rank_sub]
        
        # Part 2: Compute C[:, 4:8] = A @ B[:, 4:8]
        # U is same
        U[:, rank_sub:rank] = U_sub
        
        # V_sub index: k*4 + j
        # V_global index for block 2: k*8 + (4 + j)
        for k in range(4):
            for j in range(4):
                V[k*8 + 4 + j, rank_sub:rank] = V_sub[k*4 + j, 0:rank_sub]
        
        # W_sub index: k*4 + i
        # Global W index for block 2 (C cols 4-7): (4 + k)*4 + i
        for k in range(4):
            for i in range(4):
                W[(4 + k)*4 + i, rank_sub:rank] = W_sub[k*4 + i, 0:rank_sub]
                
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Fallback to naive if verification fails (should not happen)
        U_fb = np.zeros((n*m, n*m*p), dtype=np.float64)
        V_fb = np.zeros((m*p, n*m*p), dtype=np.float64)
        W_fb = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1 # Wait, V is m*p. Index k*p + j?
                    # V shape (m*p, rank). Index in V for B[k,j] is k*p + j.
                    # In naive: B[k,j] * A[i,k] -> C[i,j].
                    # U[i*m+k], V[k*p+j], W[j*n+i].
                    V_fb[k*p+j, col] = 1
                    W_fb[j*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    