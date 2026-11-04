"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on splitting m=3 into 1+2.
    # Part 1: A[:, 0] * B[0, :] (2x1x10) -> Rank 20
    # Part 2: A[:, 1:3] * B[1:3, :] (2x2x10) -> Rank 35 (5 blocks of 2x2x2 Strassen)
    # Total Rank = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_left (2x1) * B_top (1x10)
    # C[i, k] += A[i, 0] * B[0, k]
    # Indices:
    # U: A[i, 0] -> index i * m + 0 = i * 3 + 0
    # V: B[0, k] -> index 0 * p + k = k
    # W: C[i, k] -> index k * n + i = k * 2 + i
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] += 1
            V[0 * p + k][col] += 1
            W[k * n + i][col] += 1
            col += 1
            
    # Part 2: A_right (2x2) * B_bottom (2x10)
    # Split B_bottom (2x10) into 5 blocks of 2x2: B_j
    # For each block j (0..4), columns 2*j and 2*j+1
    # Use Strassen for 2x2x2 multiplication.
    # Strassen terms for A (2x2), B (2x2), C (2x2):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A_right indices in U:
    # A00 -> 0*3+1 = 1
    # A01 -> 0*3+2 = 2
    # A10 -> 1*3+1 = 4
    # A11 -> 1*3+2 = 5
    
    # B_bottom indices in V:
    # B00 -> 1*10 + k0
    # B01 -> 1*10 + k1
    # B10 -> 2*10 + k0
    # B11 -> 2*10 + k1
    
    # C indices in W:
    # C00 -> k0*2 + 0
    # C01 -> k1*2 + 0
    # C10 -> k0*2 + 1
    # C11 -> k1*2 + 1

    for j in range(5):
        k0 = 2 * j
        k1 = 2 * j + 1
        
        # M1: (A00 + A11) * (B00 + B11)
        U[1][col] += 1
        U[5][col] += 1
        V[1 * p + k0][col] += 1
        V[2 * p + k1][col] += 1
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B00
        U[4][col] += 1
        U[5][col] += 1
        V[1 * p + k0][col] += 1
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B01 - B11)
        U[1][col] += 1
        V[1 * p + k1][col] += 1
        V[2 * p + k1][col] -= 1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M4: A11 * (B10 - B00)
        U[5][col] += 1
        V[2 * p + k0][col] += 1
        V[1 * p + k0][col] -= 1
        W[k0 * n + 0][col] += 1
        W[k0 * n + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B11
        U[1][col] += 1
        U[2][col] += 1
        V[2 * p + k1][col] += 1
        W[k0 * n + 0][col] -= 1
        W[k1 * n + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B00 + B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B10 + B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[2 * p + k0][col] += 1
        V[2 * p + k1][col] += 1
        W[k0 * n + 0][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
