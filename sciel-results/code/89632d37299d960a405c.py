def construct():
    n, m, p = 2, 5, 5
    rank = 5
    U = [[0]*rank for _ in range(n*m)]  # 10 x 5
    V = [[0]*rank for _ in range(m*p)]  # 25 x 5
    W = [[0]*rank for _ in range(p*n)]  # 10 x 5
    
    for j in range(m):  # j = 0..4
        r = j
        for i in range(n):  # i = 0,1
            U[i*m + j][r] = 1
        for k in range(p):  # k = 0..4
            V[j*p + k][r] = 1
        for i in range(n):
            for k in range(p):
                W[k*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
