"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition:
    # Block 1: 2x2x2 Strassen (Rank 7) for i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: 2x1x3 Standard (Rank 6) for i in {0,1}, j=2, k in {0,1,2}
    # Block 3: 2x2x1 Standard (Rank 4) for i in {0,1}, j in {0,1}, k=2
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to index U: i*m + j
    # Helper to index V: j*p + k
    # Helper to index W: k*n + i
    
    # --- Block 1: Strassen (Offset 0) ---
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1 = (A00 + A11)(B00 + B11)
    c = 0
    U[0][c] = 1; U[4][c] = 1
    V[0][c] = 1; V[4][c] = 1
    W[0][c] = 1; W[3][c] = 1 # C00, C11
    
    # M2 = (A10 + A11)B00
    c = 1
    U[3][c] = 1; U[4][c] = 1
    V[0][c] = 1
    W[2][c] = 1; W[3][c] = -1 # C10, C11
    
    # M3 = A00(B01 - B11)
    c = 2
    U[0][c] = 1
    V[1][c] = 1; V[4][c] = -1
    W[1][c] = 1; W[3][c] = 1 # C01, C11
    
    # M4 = A11(B10 - B00)
    c = 3
    U[4][c] = 1
    V[3][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[2][c] = 1 # C00, C10
    
    # M5 = (A00 + A01)B11
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[4][c] = 1
    W[0][c] = -1; W[1][c] = 1 # C00, C01
    
    # M6 = (A10 - A00)(B00 + B01)
    c = 5
    U[3][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[3][c] = 1 # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    c = 6
    U[1][c] = 1; U[4][c] = -1
    V[3][c] = 1; V[4][c] = 1
    W[0][c] = 1 # C00

    # --- Block 2: Standard 2x1x3 (Offset 7) ---
    # i in {0,1}, j=2, k in {0,1,2}
    # C_ik += A_i2 B_2k
    offset = 7
    col = offset
    for i in range(2):
        for k in range(3):
            U[i * 3 + 2][col] = 1
            V[2 * 3 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1

    # --- Block 3: Standard 2x2x1 (Offset 13) ---
    # i in {0,1}, j in {0,1}, k=2
    # C_i2 += sum_j A_ij B_j2
    offset = 13
    col = offset
    for i in range(2):
        for j in range(2):
            U[i * 3 + j][col] = 1
            V[j * 3 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
