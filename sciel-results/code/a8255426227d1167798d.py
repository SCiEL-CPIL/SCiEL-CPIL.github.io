"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen part for first 2 columns of B (indices 0, 1)
    # M1 = (A11 + A22)(B11 + B22)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[6][col] = 1
    col += 1
    
    # M2 = (A21 + A22)B11
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[5][col] = 1; W[6][col] = -1
    col += 1
    
    # M3 = A11(B12 - B22)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1; W[6][col] = 1
    col += 1
    
    # M4 = A22(B21 - B11)
    U[3][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M5 = (A11 + A12)B22
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[6][col] = 1
    col += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # Schoolbook part for remaining 3 columns of B (indices 2, 3, 4)
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
