"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff):
        # u_idx, v_idx, w_idx are lists of (index, coeff)
        for ui, uc in u_idx:
            U[ui][0] += uc # Placeholder, need column
        # Actually, let's just build column by column
    
    col = 0
    
    # --- Part 1: A1 * B1a (Strassen 2x2x2) ---
    # A1: rows 0,1 of A; cols 0,1 of A. Indices: 0,1,3,4
    # B1a: rows 0,1 of B; cols 0,1 of B. Indices: 0,1,5,6
    # Output: C[0,0], C[0,1], C[1,0], C[1,1]. W indices: 0, 2, 1, 3
    
    # M1 = (A[1,1]-A[0,1]) * (B[0,0]+B[0,1])
    U[4][col] = 1; U[1][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[2][col] = 1; W[1][col] = -1
    col += 1
    
    # M2 = (A[0,0]+A[0,1]) * B[0,0]
    U[0][col] = 1; U[1][col] = 1
    V[0][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4 = A[1,1] * (B[1,0]+B[1,1])
    U[4][col] = 1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M5 = (A[0,0]-A[1,1]) * (B[0,0]+B[1,0])
    U[0][col] = 1; U[4][col] = -1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # M6 = (A[0,1]-A[1,0]) * (B[0,1]+B[1,1])
    U[1][col] = 1; U[3][col] = -1
    V[1][col] = 1; V[6][col] = 1
    W[3][col] = -1
    col += 1
    
    # M7 = (A[0,0]+A[1,0]) * B[1,1]
    U[0][col] = 1; U[3][col] = 1
    V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # --- Part 2: A1 * B1b (Strassen 2x2x2) ---
    # B1b: cols 2,3. Indices: 2,3,7,8
    # Output: C[0,2], C[0,3], C[1,2], C[1,3]. W indices: 4, 6, 5, 7
    
    # N1
    U[4][col] = 1; U[1][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[6][col] = 1; W[5][col] = -1
    col += 1
    
    # N2
    U[0][col] = 1; U[1][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # N3
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[5][col] = 1
    col += 1
    
    # N4
    U[4][col] = 1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = -1; W[5][col] = 1
    col += 1
    
    # N5
    U[0][col] = 1; U[4][col] = -1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # N6
    U[1][col] = 1; U[3][col] = -1
    V[3][col] = 1; V[8][col] = 1
    W[7][col] = -1
    col += 1
    
    # N7
    U[0][col] = 1; U[3][col] = 1
    V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # --- Part 3: A1 * B1c (Schoolbook 2x2x1) ---
    # B1c: col 4. Indices: 4, 9
    # Output: C[0,4], C[1,4]. W indices: 8, 9
    
    # K1 = A[0,0] * B[0,4]
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # K2 = A[0,1] * B[1,4]
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # K3 = A[1,0] * B[0,4]
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # K4 = A[1,1] * B[1,4]
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    # --- Part 4: A2 * B2 (Schoolbook 2x1x5) ---
    # A2: col 2. Indices: 2, 5
    # B2: row 2. Indices: 10, 11, 12, 13, 14
    # Output: C[0,0..4], C[1,0..4]. W indices: 0,2,4,6,8 and 1,3,5,7,9
    
    # L00
    U[2][col] = 1
    V[10][col] = 1
    W[0][col] = 1
    col += 1
    
    # L01
    U[2][col] = 1
    V[11][col] = 1
    W[2][col] = 1
    col += 1
    
    # L02
    U[2][col] = 1
    V[12][col] = 1
    W[4][col] = 1
    col += 1
    
    # L03
    U[2][col] = 1
    V[13][col] = 1
    W[6][col] = 1
    col += 1
    
    # L04
    U[2][col] = 1
    V[14][col] = 1
    W[8][col] = 1
    col += 1
    
    # L10
    U[5][col] = 1
    V[10][col] = 1
    W[1][col] = 1
    col += 1
    
    # L11
    U[5][col] = 1
    V[11][col] = 1
    W[3][col] = 1
    col += 1
    
    # L12
    U[5][col] = 1
    V[12][col] = 1
    W[5][col] = 1
    col += 1
    
    # L13
    U[5][col] = 1
    V[13][col] = 1
    W[7][col] = 1
    col += 1
    
    # L14
    U[5][col] = 1
    V[14][col] = 1
    W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
