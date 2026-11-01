def _rank32_decomposition():
    n, m, p = 2, 2, 16
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    # j=0 part: r = 0..15
    for r in range(16):
        k = r
        U[0][r] = 1  # i=0, j=0
        U[2][r] = 1  # i=1, j=0
        V[k][r] = 1  # j=0, k
        W[2*k][r] = 1  # k, i=0 -> 2k+0 = 2k
        
    # j=1 part: r = 16..31
    for r in range(16, 32):
        k = r - 16
        U[1][r] = 1  # i=0, j=1
        U[3][r] = 1  # i=1, j=1
        V[16 + k][r] = 1  # j=1, k -> 1*16+k
        W[2*k + 1][r] = 1  # k, i=1 -> 2k+1
    
    return U, V, W
