def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    rank = 2 * p  # 24
    U = [[0] * rank for _ in range(n * m)]   # 4 x 24
    V = [[0] * rank for _ in range(m * p)]   # 24 x 24
    W = [[0] * rank for _ in range(p * n)]   # 24 x 24
    
    for k in range(p):
        for j in range(m):
            c = 2 * k + j
            # U: u_c[j] = 1, u_c[2+j] = 1
            U[j][c] = 1
            U[m + j][c] = 1
            # V: v_c[j*p+k] = 1
            V[j * p + k][c] = 1
            # W: w_c[k] = 1, w_c[2k+1] = 1
            W[k * n + 0][c] = 1
            W[k * n + 1][c] = 1
    
    return {"U": U, "V": V, "W": W}
