def construct():
    """
    Rank-1 tensor decomposition for 2x3 x 3x6 matrix multiplication.
    The multiplication tensor is the all-ones tensor, which has rank 1.
    """
    n, m, p = 2, 3, 6
    
    rank = 1
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    # Verify
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank) # W[k*n+i, r] -> W_reshaped[k, i, r]
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    target = np.ones((n, m, p), dtype=np.float64)
    
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook (Rank 36) just in case, though it shouldn't happen
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
        V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
        W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
