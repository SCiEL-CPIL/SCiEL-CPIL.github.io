def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    N, M, P = 2, 4, 6
    NM = N * M  # 8
    MP = M * P  # 24
    PN = P * N  # 12
    R = 24  # Lower bound is max(NM, MP, PN) = 24
    
    U = np.zeros((NM, R), dtype=np.int64)
    V = np.zeros((MP, R), dtype=np.int64)
    W = np.zeros((PN, R), dtype=np.int64)
    
    for c in range(R):
        j_c = c // P
        k_c = c % P
        
        # V[j, k, c] is 1 if j==j_c and k==k_c
        V[j_c * P + k_c, c] = 1
        
        # U[i, j, c] is 1 for all i, if j==j_c
        for i in range(N):
            U[i * M + j_c, c] = 1
            
        # W[k, i, c] is 1 for all i, if k==k_c
        for i in range(N):
            W[k_c * N + i, c] = 1
            
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
