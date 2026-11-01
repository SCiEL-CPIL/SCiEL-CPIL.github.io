"""Matrix multiplication 2x3 * 3x8 tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[p + k1][col] = 1
        W[k0][col] = 1
        W[n + k1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        W[n + k0][col] = 1
        W[n + k1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[k1][col] = 1
        V[p + k1][col] = -1
        W[k1][col] = 1
        W[n + k1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][col] = 1
        V[p + k0][col] = 1
        V[k0][col] = -1
        W[k0][col] = 1
        W[n + k0][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1
        U[1][col] = 1
        V[p + k1][col] = 1
        W[k0][col] = -1
        W[k1][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[k1][col] = 1
        W[n + k1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1
        U[4][col] = -1
        V[p + k0][col] = 1
        V[p + k1][col] = 1
        W[k0][col] = 1
        col += 1
        
        # P1 = A02 * B2k0
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0][col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1][col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[n + k0][col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[n + k1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
