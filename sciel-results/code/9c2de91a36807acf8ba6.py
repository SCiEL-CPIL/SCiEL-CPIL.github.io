"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # M1: (A00 + A11)(B0c0 + B1c1)
        U[0][col] = 1
        U[3][col] = 1
        V[c0][col] = 1
        V[8 + c1][col] = 1
        W[0][col] = 1
        W[3][col] = 1
        col += 1
        
        # M2: (A10 + A11)B0c0
        U[2][col] = 1
        U[3][col] = 1
        V[c0][col] = 1
        W[1][col] = 1
        W[3][col] = -1
        col += 1
        
        # M3: A00(B1c0 - B1c1)
        U[0][col] = 1
        V[8 + c0][col] = 1
        V[8 + c1][col] = -1
        W[2][col] = 1
        W[3][col] = 1
        col += 1
        
        # M4: A11(B0c0 - B0c1)
        U[3][col] = 1
        V[c0][col] = 1
        V[c1][col] = -1
        W[0][col] = 1
        W[1][col] = 1
        col += 1
        
        # M5: (A00 + A01)B1c1
        U[0][col] = 1
        U[1][col] = 1
        V[8 + c1][col] = 1
        W[0][col] = -1
        W[2][col] = 1
        col += 1
        
        # M6: (A10 - A00)(B0c0 + B0c1)
        U[2][col] = 1
        U[0][col] = -1
        V[c0][col] = 1
        V[c1][col] = 1
        W[3][col] = 1
        col += 1
        
        # M7: (A01 - A11)(B1c0 + B1c1)
        U[1][col] = 1
        U[3][col] = -1
        V[8 + c0][col] = 1
        V[8 + c1][col] = 1
        W[0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
