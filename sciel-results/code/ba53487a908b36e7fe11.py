    U = [[0.0] * rank for _ in range(size_U)]
    V = [[0.0] * rank for _ in range(size_V)]
    W = [[0.0] * rank for _ in range(size_W)]
    
    for r in range(rank):
        for i in range(size_U):
            U[i][r] = U_cols[r][i]
        for j in range(size_V):
            V[j][r] = V_cols[r][j]
        for k in range(size_W):
            W[k][r] = W_cols[r][k]
