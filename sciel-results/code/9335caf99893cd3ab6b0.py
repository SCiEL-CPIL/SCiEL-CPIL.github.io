def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = n * m  # 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(n):  # a = i index
        for b in range(m):  # b = j index
            # U[i*m+j, col] = 1 iff i=a, j=b
            U[a * m + b][col] = 1
            # V[j'*p+k, col] = 1 iff j'=b (for all k)
            for k in range(p):
                V[b * p + k][col] = 1
            # W[k'*n+i', col] = 1 iff i'=a (for all k')
            for kp in range(p):
                W[kp * n + a][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
