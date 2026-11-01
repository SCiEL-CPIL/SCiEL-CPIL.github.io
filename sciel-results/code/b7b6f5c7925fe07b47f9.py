"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Strassen's algorithm for 2x2 blocks
    # 4 blocks of 2 columns each, 7 Strassen products per block
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes: C00 +1, C11 +1
        U[0][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # Contributes: C10 +1
        U[2][col] = 1; U[3][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        # Contributes: C01 +1, C11 +1
        U[0][col] = 1
        V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        # Contributes: C00 +1, C10 +1
        U[3][col] = 1
        V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        # Contributes: C01 +1, C11 -1
        U[0][col] = 1; U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes: C11 +1
        U[2][col] = 1; U[0][col] = -1
        V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes: C00 +1
        U[1][col] = 1; U[3][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
