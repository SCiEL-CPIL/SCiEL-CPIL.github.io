"""Matrix multiplication 2x3 by 3x8 with rank-44 decomposition."""
DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j=0, schoolbook 2x1 * 1x8, rank 16
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: Strassen 2x2 * 2x8 in 4 column-blocks, rank 28
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        # M1: (A00+A11)*(B00+B11) -> C00+1, C11+1
        U[0*m+1][col] = 1; U[1*m+2][col] = 1
        V[1*p+k0][col] = 1; V[2*p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        # M2: (A10+A11)*B00 -> C10+1, C11-1
        U[1*m+1][col] = 1; U[1*m+2][col] = 1
        V[1*p+k0][col] = 1
        W[k0*n+1][col] = 1; W[k1*n+1][col] = -1
        col += 1
        # M3: A00*(B01-B11) -> C01+1, C11+1
        U[0*m+1][col] = 1
        V[1*p+k1][col] = 1; V[2*p+k1][col] = -1
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        # M4: A11*(B10-B00) -> C00+1, C10+1
        U[1*m+2][col] = 1
        V[2*p+k0][col] = 1; V[1*p+k0][col] = -1
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1
        col += 1
        # M5: (A00+A10)*B11 -> C00-1, C01+1
        U[0*m+1][col] = 1; U[1*m+1][col] = 1
        V[2*p+k1][col] = 1
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1
        col += 1
        # M6: (A11-A10)*(B00+B10) -> C11+1
        U[1*m+2][col] = 1; U[1*m+1][col] = -1
        V[1*p+k0][col] = 1; V[2*p+k0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        # M7: (A01-A11)*(B01+B11) -> C00+1
        U[0*m+2][col] = 1; U[1*m+2][col] = -1
        V[1*p+k1][col] = 1; V[2*p+k1][col] = 1
        W[k0*n+0][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
