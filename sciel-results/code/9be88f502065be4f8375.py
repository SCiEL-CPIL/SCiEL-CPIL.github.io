def construct():
    n, m, p = 2, 2, 7
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        # Block for column k of B
        # Terms: A00*B0k, A01*B1k, A10*B0k, A11*B1k
        # i=0, j=0 -> U[0], V[0*p+k], W[k*n+0]
        U[0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # i=0, j=1 -> U[1], V[1*p+k], W[k*n+0]
        U[1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # i=1, j=0 -> U[2], V[0*p+k], W[k*n+1]
        U[2][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # i=1, j=1 -> U[3], V[1*p+k], W[k*n+1]
        U[3][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
