def get_rank_6():
    n, m, p = 2, 3, 9
    rank = 6
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U[i*m + j, r] = 1
            U[i * m + j][r] = 1
            # V[j'*p + l, r] = delta_{j', j} for all l
            for l in range(p):
                V[j * p + l][r] = 1
            # W[l*n + i, r] = 1 for all l
            for l in range(p):
                W[l * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
