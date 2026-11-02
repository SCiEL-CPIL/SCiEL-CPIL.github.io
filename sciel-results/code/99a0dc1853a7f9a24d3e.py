def construct():
    n, m, p = DIMENSIONS
    
    U_S, V_S, W_S = _block_strassen_4x4x4()
    
    U_SB = np.zeros((n*m, n*m), dtype=int)
    V_SB = np.zeros((m*p, n*m), dtype=int)
    W_SB = np.zeros((p*n, n*m), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            U_SB[i*m + j, col] = 1
            V_SB[j*p + (p-1), col] = 1
            W_SB[(p-1)*n + i, col] = 1
            col += 1
            
    U_full = np.hstack([U_S, U_SB])
    
    V_S_padded = np.zeros((m*p, 49), dtype=int)
    for j in range(m):
        for k in range(p-1):
            idx_S = j * (p-1) + k
            idx_full = j * p + k
            V_S_padded[idx_full, :] = V_S[idx_S, :]
            
    V_full = np.hstack([V_S_padded, V_SB])
    
    W_S_padded = np.zeros((p*n, 49), dtype=int)
    W_S_padded[: (p-1)*n, :] = W_S
    
    W_full = np.hstack([W_S_padded, W_SB])
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
