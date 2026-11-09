"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen parameters for 2x2 blocks
    # A_T is A[0:2, 0:2]
    # B_L is B[0:2, :]
    # We process 4 blocks of B_L, each 2x2
    
    # Strassen product definitions
    # P1: (A00 + A11) * (B00 + B11)
    # P2: (A10 + A11) * B00
    # P3: A00 * (B01 - B11)
    # P4: A11 * (B10 - B00)
    # P5: (A00 + A01) * B11
    # P6: (A10 - A00) * (B00 + B01)
    # P7: (A01 - A11) * (B10 + B11)
    
    # Output mappings
    # M00: P1 + P4 - P5 - P7
    # M01: P3 + P5
    # M10: P2 + P4
    # M11: P1 - P3 + P6 + P7
    
    # Indices in A for A_T:
    # A00 -> (0,0) -> idx 0
    # A01 -> (0,1) -> idx 1
    # A10 -> (1,0) -> idx 3
    # A11 -> (1,1) -> idx 4
    
    # Indices in B for B_L block b (cols 2b, 2b+1):
    # B00 -> (0, 2b) -> idx 0*p + 2b
    # B01 -> (0, 2b+1) -> idx 0*p + 2b+1
    # B10 -> (1, 2b) -> idx 1*p + 2b
    # B11 -> (1, 2b+1) -> idx 1*p + 2b+1
    
    # Indices in C for block b (rows 0,1; cols 2b, 2b+1):
    # C00 -> (2b, 0) -> idx 2b*n + 0
    # C01 -> (2b+1, 0) -> idx (2b+1)*n + 0
    # C10 -> (2b, 1) -> idx 2b*n + 1
    # C11 -> (2b+1, 1) -> idx (2b+1)*n + 1
    
    for b in range(4):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # B indices
        b00_idx = 0 * p + c0
        b01_idx = 0 * p + c1
        b10_idx = 1 * p + c0
        b11_idx = 1 * p + c1
        
        # C indices
        c00_idx = c0 * n + 0
        c01_idx = c1 * n + 0
        c10_idx = c0 * n + 1
        c11_idx = c1 * n + 1
        
        # P1
        U[0][col] += 1
        U[4][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # P2
        U[3][col] += 1
        U[4][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P3
        U[0][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] += -1
        W[c01_idx][col] += 1
        col += 1
        
        # P4
        U[4][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] += -1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        # P5
        U[0][col] += 1
        U[1][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += -1
        W[c01_idx][col] += 1
        col += 1
        
        # P6
        U[3][col] += 1
        U[0][col] += -1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # P7
        U[1][col] += 1
        U[4][col] += -1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += -1
        W[c11_idx][col] += 1
        col += 1

    # Remaining terms: A_R B_R
    # A_R is A[0:2, 2] -> idx 2, 5
    # B_R is B[2, :] -> idx 2*p + k
    # C[0, k] += A[0, 2] B[2, k]
    # C[1, k] += A[1, 2] B[2, k]
    for k in range(p):
        # Term for A[0,2] B[2,k]
        U[2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term for A[1,2] B[2,k]
        U[5][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # Remaining terms: A_B B_L
    # A_B is A[2, 0:2] -> idx 6, 7
    # B_L is B[0:2, :]
    # C[2, k] += A[2, 0] B[0, k] + A[2, 1] B[1, k]
    for k in range(p):
        # Term for A[2,0] B[0,k]
        U[6][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
        # Term for A[2,1] B[1,k]
        U[7][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    # Remaining terms: A_BR B_R
    # A_BR is A[2, 2] -> idx 8
    # B_R is B[2, :]
    # C[2, k] += A[2, 2] B[2, k]
    for k in range(p):
        U[8][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
