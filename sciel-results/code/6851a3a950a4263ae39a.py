"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A_L B_L (Rank 1)
    # A col 0, B row 0
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][col] = 1
    col += 1
    
    # 2. Strassen Block 1 (A_R, B_R1)
    # A cols 1,2. B rows 1,2, cols 0,1.
    
    # M1: (A01+A12)*(B10+B21)
    U[0*m + 1][col] = 1
    U[1*m + 2][col] = 1
    V[1*p + 0][col] = 1
    V[2*p + 1][col] = 1
    W[0*n + 0][col] = 1
    W[1*n + 1][col] = 1
    col += 1
    
    # M2: (A11+A12)*B21
    U[1*m + 1][col] = 1
    U[1*m + 2][col] = 1
    V[2*p + 1][col] = 1
    W[1*n + 0][col] = 1
    W[1*n + 1][col] = -1
    col += 1
    
    # M3: A01*(B11-B21)
    U[0*m + 1][col] = 1
    V[1*p + 1][col] = 1
    V[2*p + 1][col] = -1
    W[0*n + 1][col] = 1
    W[1*n + 1][col] = 1
    col += 1
    
    # M4: A12*(B20-B10)
    U[1*m + 2][col] = 1
    V[2*p + 0][col] = 1
    V[1*p + 0][col] = -1
    W[0*n + 0][col] = 1
    W[1*n + 0][col] = 1
    col += 1
    
    # M5: (A01+A02)*B10
    U[0*m + 1][col] = 1
    U[0*m + 2][col] = 1
    V[1*p + 0][col] = 1
    W[0*n + 0][col] = -1
    W[0*n + 1][col] = 1
    col += 1
    
    # M6: (A11-A01)*(B10+B11)
    U[1*m + 1][col] = 1
    U[0*m + 1][col] = -1
    V[1*p + 0][col] = 1
    V[1*p + 1][col] = 1
    W[1*n + 1][col] = 1
    col += 1
    
    # M7: (A02-A12)*(B20+B21)
    U[0*m + 2][col] = 1
    U[1*m + 2][col] = -1
    V[2*p + 0][col] = 1
    V[2*p + 1][col] = 1
    W[0*n + 0][col] = 1
    W[1*n + 0][col] = 1
    col += 1
    
    # 3. Strassen Block 2 (A_R, B_R2)
    # A cols 1,2. B rows 1,2, cols 2,3.
    
    # M1: (A01+A12)*(B12+B23)
    U[0*m + 1][col] = 1
    U[1*m + 2][col] = 1
    V[1*p + 2][col] = 1
    V[2*p + 3][col] = 1
    W[2*n + 0][col] = 1
    W[3*n + 1][col] = 1
    col += 1
    
    # M2: (A11+A12)*B23
    U[1*m + 1][col] = 1
    U[1*m + 2][col] = 1
    V[2*p + 3][col] = 1
    W[2*n + 1][col] = 1
    W[3*n + 1][col] = -1
    col += 1
    
    # M3: A01*(B13-B23)
    U[0*m + 1][col] = 1
    V[1*p + 3][col] = 1
    V[2*p + 3][col] = -1
    W[3*n + 0][col] = 1
    W[3*n + 1][col] = 1
    col += 1
    
    # M4: A12*(B22-B12)
    U[1*m + 2][col] = 1
    V[2*p + 2][col] = 1
    V[1*p + 2][col] = -1
    W[2*n + 0][col] = 1
    W[2*n + 1][col] = 1
    col += 1
    
    # M5: (A01+A02)*B12
    U[0*m + 1][col] = 1
    U[0*m + 2][col] = 1
    V[1*p + 2][col] = 1
    W[2*n + 0][col] = -1
    W[3*n + 0][col] = 1
    col += 1
    
    # M6: (A11-A01)*(B12+B13)
    U[1*m + 1][col] = 1
    U[0*m + 1][col] = -1
    V[1*p + 2][col] = 1
    V[1*p + 3][col] = 1
    W[3*n + 1][col] = 1
    col += 1
    
    # M7: (A02-A12)*(B22+B23)
    U[0*m + 2][col] = 1
    U[1*m + 2][col] = -1
    V[2*p + 2][col] = 1
    V[2*p + 3][col] = 1
    W[2*n + 0][col] = 1
    W[2*n + 1][col] = 1
    col += 1
    
    # 4. Block 3 (A_R, B_R3)
    # A cols 1,2. B rows 1,2, col 4.
    
    # M1: A01 * B14
    U[0*m + 1][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 0][col] = 1
    col += 1
    
    # M2: A02 * B24
    U[0*m + 2][col] = 1
    V[2*p + 4][col] = 1
    W[4*n + 0][col] = 1
    col += 1
    
    # M3: A11 * B14
    U[1*m + 1][col] = 1
    V[1*p + 4][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    # M4: A12 * B24
    U[1*m + 2][col] = 1
    V[2*p + 4][col] = 1
    W[4*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
