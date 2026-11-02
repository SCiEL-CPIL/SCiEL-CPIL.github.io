def construct():
    n, m, p = 4, 4, 6
    rank = 16
    U = [[0]*rank for _ in range(n*m)]  # 16 x 16
    V = [[0]*rank for _ in range(m*p)]  # 24 x 16
    W = [[0]*rank for _ in range(p*n)]  # 24 x 16
    
    col = 0
    for strip in range(2):
        i_base = strip * 2
        for i_offset in range(2):
            i = i_base + i_offset
            for j in range(m):
                # 1x4x6 decomposition for row i
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
