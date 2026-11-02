def construct():
    n, m, p = 2, 2, 13
    # Standard multiplication requires 4 multiplications per column
    # Total rank = 4 * p
    rank = 4 * p
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 52)
    V = [[0] * rank for _ in range(m * p)]  # (26, 52)
    W = [[0] * rank for _ in range(p * n)]  # (26, 52)
    
    # We need to decompose C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # Term index r corresponds to (i, j, k)
    # r = i * (2 * p) + j * p + k ?
    # Let's just loop i, j, k
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index for A[i,j] is i*m + j
                U[i * m + j][r] = 1
                
                # V index for B[j,k] is j*p + k
                V[j * p + k][r] = 1
                
                # W index for C[i,k] is k*n + i (trace convention)
                W[k * n + i][r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
