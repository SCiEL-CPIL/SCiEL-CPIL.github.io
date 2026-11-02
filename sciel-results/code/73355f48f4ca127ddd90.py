def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try the rank-40 decomposition (2 * rank-20 4x4x4)
    try:
        dec_444 = _get_4x4_decomposition()
        U_444 = np.array(dec_444["U"])
        V_444 = np.array(dec_444["V"])
        W_444 = np.array(dec_444["W"])
        
        rank_444 = U_444.shape[1]
        rank_total = 2 * rank_444
        
        U = np.zeros((n * m, rank_total), dtype=float)
        V = np.zeros((m * p, rank_total), dtype=float)
        W = np.zeros((p * n, rank_total), dtype=float)
        
        # First half: columns 0-3 of B
        # U: rows 0-7 (i=0,1; j=0-3)
        U[:, 0:rank_444] = U_444[0:8, :]
        # V: rows 0-15 (j=0-3; k=0-3)
        V[0:16, 0:rank_444] = V_444[0:16, :]
        # W: rows 0-7 (k=0-3; i=0,1)
        # W_444 indices are 4*k + i. We need 2*k + i.
        for k in range(4):
            for i in range(2):
                W[k * n + i, 0:rank_444] = W_444[k * 4 + i, 0:rank_444]
                
        # Second half: columns 4-7 of B
        # U: same
        U[:, rank_444:2*rank_444] = U_444[0:8, :]
        # V: rows 0-15 (j=0-3; k=4-7)
        # V_444 indices are 4*j + k. We need 8*j + (k+4).
        for j in range(4):
            for k in range(4):
                V[j * p + k + 4, rank_444:2*rank_444] = V_444[j * 4 + k, 0:rank_444]
        # W: same
        for k in range(4):
            for i in range(2):
                W[k * n + i, rank_444:2*rank_444] = W_444[k * 4 + i, 0:rank_444]
        
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
    except Exception:
        pass
        
    # Fallback to Strassen (rank 56)
    try:
        result = _strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
    except Exception:
        pass
        
    return _schoolbook_decomposition()
