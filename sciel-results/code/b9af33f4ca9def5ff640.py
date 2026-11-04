def construct():
    n, m, p = 2, 4, 5
    rank = m * p  # 20
    
    U = [[0.0] * rank for _ in range(n * m)]  # (8, 20)
    V = [[0.0] * rank for _ in range(m * p)]  # (20, 20)
    W = [[0.0] * rank for _ in range(p * n)]  # (10, 20)
    
    for b in range(m):      # j_A = j_B
        for c in range(p):  # k_B = k_C
            r = b * p + c
            for a in range(n):  # i_A = i_C
                U[a * m + b][r] = 1.0
                W[c * n + a][r] = 1.0
            V[b * p + c][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
