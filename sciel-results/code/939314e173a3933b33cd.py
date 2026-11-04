def construct():
    n, m, p = 3, 3, 6
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Two blocks
    for block in range(2):
        base_col = block * 9
        for i in range(n):
            for j in range(m):
                for k in range(m): # m=3
                    col = base_col + i * m + j
                    # U index: i*m + j
                    U[i * m + j][col] = 1
                    # V index: j*p + (block*m + k)
                    V[j * p + block * m + k][col] = 1
                    # W index: (block*m + k)*n + i
                    W[(block * m + k) * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
