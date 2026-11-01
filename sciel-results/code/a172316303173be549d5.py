"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # M_{2,2,8} decomposed into 4 blocks of M_{2,2,2} (Strassen rank 7) -> 28
    # M_{2,1,8} decomposed into rank 2
    # Total rank = 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: j, j' in {0, 1} ---
    # Corresponds to the 2x2 sub-matrix of A and B.
    # k is split into 4 blocks of size 2.
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        base = 7 * b
        
        # M1: (A00 + A11)(B00 + B11)
        r = base
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + k0][r] = 1
        V[1 * p + k1][r] = 1
        W[k0 * n + 0][r] = 1
        W[k1 * n + 1][r] = 1
        
        # M2: (A01 + A11)B00
        r = base + 1
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + k0][r] = 1
        W[k1 * n + 1][r] = -1
        
        # M3: A00(B01 - B11)
        r = base + 2
        U[0 * m + 0][r] = 1
        V[0 * p + k1][r] = 1
        V[1 * p + k1][r] = -1
        W[k1 * n + 1][r] = 1
        
        # M4: A11(B10 - B00)
        r = base + 3
        U[1 * m + 1][r] = 1
        V[1 * p + k0][r] = 1
        V[0 * p + k0][r] = -1
        W[k0 * n + 0][r] = 1
        
        # M5: (A00 + A01)B11
        r = base + 4
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + k1][r] = 1
        W[k0 * n + 0][r] = -1
        
        # M6: (A10 - A00)(B00 + B01)
        r = base + 5
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + k0][r] = 1
        V[0 * p + k1][r] = 1
        W[k1 * n + 1][r] = 1
        
        # M7: (A01 - A11)(B10 + B11)
        r = base + 6
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + k0][r] = 1
        V[1 * p + k1][r] = 1
        W[k0 * n + 0][r] = 1

    # --- Part 2: j = 2, j' = 2 ---
    # Corresponds to the 2x1 sub-matrix of A and B.
    # Rank 2 decomposition.
    
    # Column 28: A02 * B2k -> C0k
    r = 28
    U[0 * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
        W[k * n + 0][r] = 1
    
    # Column 29: A12 * B2k -> C1k
    r = 29
    U[1 * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
        W[k * n + 1][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
