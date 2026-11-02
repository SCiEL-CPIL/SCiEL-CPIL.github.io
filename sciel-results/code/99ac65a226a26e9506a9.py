"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 * 7 = 21
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 block
    # k0, k1 are the column indices in B (and C)
    # A indices: 00, 01, 10, 11
    # B indices: j*p + k
    
    def add_strassen_block(col_offset, k0, k1):
        # M1: (A00+A11) * (B0k0+B1k0)
        U[0][col_offset + 0] = 1
        U[3][col_offset + 0] = 1
        V[0*p + k0][col_offset + 0] = 1
        V[1*p + k0][col_offset + 0] = 1
        W[k0*n + 0][col_offset + 0] = 1 # C0k0
        W[k1*n + 1][col_offset + 0] = 1 # C1k1
        
        # M2: (A10+A11) * B0k0
        U[2][col_offset + 1] = 1
        U[3][col_offset + 1] = 1
        V[0*p + k0][col_offset + 1] = 1
        W[k0*n + 1][col_offset + 1] = 1 # C1k0
        W[k1*n + 1][col_offset + 1] = -1 # C1k1
        
        # M3: A00 * (B0k1-B1k1)
        U[0][col_offset + 2] = 1
        V[0*p + k1][col_offset + 2] = 1
        V[1*p + k1][col_offset + 2] = -1
        W[k1*n + 0][col_offset + 2] = 1 # C0k1
        W[k1*n + 1][col_offset + 2] = 1 # C1k1
        
        # M4: A11 * (B1k1-B0k1)
        U[3][col_offset + 3] = 1
        V[1*p + k1][col_offset + 3] = 1
        V[0*p + k1][col_offset + 3] = -1
        W[k0*n + 0][col_offset + 3] = 1 # C0k0
        W[k0*n + 1][col_offset + 3] = 1 # C1k0
        
        # M5: (A00+A01) * B1k1
        U[0][col_offset + 4] = 1
        U[1][col_offset + 4] = 1
        V[1*p + k1][col_offset + 4] = 1
        W[k0*n + 0][col_offset + 4] = -1 # C0k0
        W[k1*n + 0][col_offset + 4] = 1  # C0k1
        
        # M6: (A10-A00) * (B0k0+B0k1)
        U[2][col_offset + 5] = 1
        U[0][col_offset + 5] = -1
        V[0*p + k0][col_offset + 5] = 1
        V[0*p + k1][col_offset + 5] = 1
        W[k1*n + 1][col_offset + 5] = 1 # C1k1
        
        # M7: (A01-A11) * (B1k0+B1k1)
        U[1][col_offset + 6] = 1
        U[3][col_offset + 6] = -1
        V[1*p + k0][col_offset + 6] = 1
        V[1*p + k1][col_offset + 6] = 1
        W[k0*n + 0][col_offset + 6] = 1 # C0k0

    add_strassen_block(0, 0, 1)
    add_strassen_block(7, 2, 3)
    add_strassen_block(14, 4, 5)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
