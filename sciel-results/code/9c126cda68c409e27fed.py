"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # P1: A00 * (B0k1 - B1k1) -> C0k1 + C1k1
        U[0][col] = 1
        V[k1][col] = 1
        V[10 + k1][col] = -1
        W[2 * k1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # P2: (A00 + A01) * B1k1 -> -C0k0
        U[0][col] = 1
        U[1][col] = 1
        V[10 + k1][col] = 1
        W[2 * k0][col] = -1
        col += 1
        
        # P3: A10 * (B0k0 + B1k0) -> C1k0 - C1k1
        U[3][col] = 1
        V[k0][col] = 1
        V[10 + k0][col] = 1
        W[2 * k0 + 1][col] = 1
        W[2 * k1 + 1][col] = -1
        col += 1
        
        # P4: A11 * (B1k0 - B0k0) -> C0k0 + C1k1
        U[4][col] = 1
        V[10 + k0][col] = 1
        V[k0][col] = -1
        W[2 * k0][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # P5: (A00 + A11) * (B0k0 + B1k1) -> C0k0 + C0k1 + C1k1
        U[0][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[10 + k1][col] = 1
        W[2 * k0][col] = 1
        W[2 * k1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1
        
        # P6: (A01 - A00) * (B0k0 + B1k0) -> C0k0
        U[1][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[10 + k0][col] = 1
        W[2 * k0][col] = 1
        col += 1
        
        # P7: (A10 - A11) * (B0k1 - B1k1) -> C1k0
        U[3][col] = 1
        U[4][col] = -1
        V[k1][col] = 1
        V[10 + k1][col] = -1
        W[2 * k0 + 1][col] = 1
        col += 1
        
        # Q1: A02 * B2k0 -> C0k0
        U[2][col] = 1
        V[20 + k0][col] = 1
        W[2 * k0][col] = 1
        col += 1
        
        # Q2: A02 * B2k1 -> C0k1
        U[2][col] = 1
        V[20 + k1][col] = 1
        W[2 * k1][col] = 1
        col += 1
        
        # Q3: A12 * B2k0 -> C1k0
        U[5][col] = 1
        V[20 + k0][col] = 1
        W[2 * k0 + 1][col] = 1
        col += 1
        
        # Q4: A12 * B2k1 -> C1k1
        U[5][col] = 1
        V[20 + k1][col] = 1
        W[2 * k1 + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
