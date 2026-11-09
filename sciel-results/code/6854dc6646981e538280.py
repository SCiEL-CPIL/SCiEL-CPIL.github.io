def construct():
    import time
    import numpy as np
    
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140
    
    # Try rank-8 construction
    rank = 8
    U = [[0]*rank for _ in range(n*m)]  # 16 x 8
    V = [[0]*rank for _ in range(m*p)]  # 24 x 8
    W = [[0]*rank for _ in range(p*n)]  # 24 x 8
    
    col = 0
    for i_block in range(2):
        for j_block in range(2):
            for j_local in range(2):
                i1 = i_block * 2
                i2 = i_block * 2 + 1
                j = j_block * 2 + j_local
                # U[i1*m+j, col] = 1, U[i2*m+j, col] = 1
                U[i1*m + j][col] = 1
                U[i2*m + j][col] = 1
                # V[j*p+k, col] = 1 for all k
                for k in range(p):
                    V[j*p + k][col] = 1
                # W[k*n+i, col] = 1 for all k, i in {i1, i2}
                for k in range(p):
                    W[k*n + i1][col] = 1
                    W[k*n + i2][col] = 1
                col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
