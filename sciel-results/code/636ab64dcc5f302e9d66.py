def construct():
    import numpy as np
    n, m, p = 2, 2, 13
    rank = 46
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 7 terms for 2x2 x 2x2
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1], # M7
    ]
    strassen_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,0,1,-1], # M3
        [0,1,0,0],  # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1],  # M7
    ]
    strassen_W = [
        [1,0,0,1],  # M1
        [0,0,1,0],  # M2
        [0,1,0,0],  # M3
        [-1,0,1,0], # M4
        [-1,1,0,0], # M5
        [0,0,0,1],  # M6
        [1,0,0,-1], # M7
    ]
    
    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(strassen_U[t][i*2+j] * strassen_V[t][j*2+k] * strassen_W[t][k*2+i] for t in range(7))
                assert abs(val - 1.0) < 1e-9, f"2x2x2 failed at ({i},{j},{k}): {val}"
    
    # Now build 2x2 x 2x13
    t = 0
    for c in range(0, 12, 2):
        for tt in range(7):
            # U
            for i in range(2):
                for j in range(2):
                    U[i*2+j, t+tt] = strassen_U[tt][i*2+j]
            # V: local B indices 0,1,2,3 → global j*13+c+k_local
            for j in range(2):
                for k_local in range(2):
                    V[j*13+c+k_local, t+tt] = strassen_V[tt][j*2+k_local]
            # W: local C indices 0,1,2,3 → global k*2+i
            for k_local in range(2):
                for i in range(2):
                    W[(c+k_local)*2+i, t+tt] = strassen_W[tt][k_local*2+i]
        t += 7
    
    # Schoolbook for column 12
    for jj in range(4):
        i = jj // 2
        j = jj % 2
        U[i*2+j, t+jj] = 1
        V[j*13+12, t+jj] = 1
        W[12*2+i, t+jj] = 1
    t += 4
    
    # Full verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                assert abs(val - 1.0) < 1e-9, f"Failed at ({i},{j},{k}): {val}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
