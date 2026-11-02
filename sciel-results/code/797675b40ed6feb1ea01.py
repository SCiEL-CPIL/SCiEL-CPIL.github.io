import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for 2x2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11, start_col):
        # 1. (a+d)(e+h)
        U[a00][start_col] += 1
        U[a10][start_col] += 1
        V[b00][start_col] += 1
        V[b11][start_col] += 1
        W[c00][start_col] += 1
        W[c11][start_col] += 1
        
        # 2. (c+d)e
        U[a10][start_col+1] += 1
        U[a11][start_col+1] += 1
        V[b00][start_col+1] += 1
        W[c10][start_col+1] += 1
        W[c11][start_col+1] -= 1
        
        # 3. a(f-h)
        U[a00][start_col+2] += 1
        V[b01][start_col+2] += 1
        V[b10][start_col+2] -= 1
        W[c01][start_col+2] += 1
        W[c11][start_col+2] += 1
        
        # 4. d(g-e)
        U[a11][start_col+3] += 1
        V[b10][start_col+3] += 1
        V[b00][start_col+3] -= 1
        W[c00][start_col+3] += 1
        W[c10][start_col+3] += 1
        
        # 5. (a+b)h
        U[a00][start_col+4] += 1
        U[a01][start_col+4] += 1
        V[b11][start_col+4] += 1
        W[c00][start_col+4] -= 1
        W[c01][start_col+4] += 1
        
        # 6. (c-a)(e+f)
        U[a10][start_col+5] += 1
        U[a00][start_col+5] -= 1
        V[b00][start_col+5] += 1
        V[b01][start_col+5] += 1
        W[c11][start_col+5] += 1
        
        # 7. (b-d)(g+h)
        U[a01][start_col+6] += 1
        U[a11][start_col+6] -= 1
        V[b10][start_col+6] += 1
        V[b11][start_col+6] += 1
        W[c00][start_col+6] += 1

    # A indices for A1 (cols 0,1)
    a1_00, a1_01, a1_10, a1_11 = 0, 1, 2, 3
    # A indices for A2 (cols 2,3)
    a2_00, a2_01, a2_10, a2_11 = 4, 5, 6, 7
    
    for j in range(3):
        # B1,j indices (rows 0,1; cols 2j, 2j+1)
        b1_00 = 0 * 7 + 2 * j
        b1_01 = 0 * 7 + 2 * j + 1
        b1_10 = 1 * 7 + 2 * j
        b1_11 = 1 * 7 + 2 * j + 1
        
        # B2,j indices (rows 2,3; cols 2j, 2j+1)
        b2_00 = 2 * 7 + 2 * j
        b2_01 = 2 * 7 + 2 * j + 1
        b2_10 = 3 * 7 + 2 * j
        b2_11 = 3 * 7 + 2 * j + 1
        
        # C indices (rows 0,1; cols 2j, 2j+1)
        c_00 = (2 * j) * 2 + 0
        c_01 = (2 * j) * 2 + 1
        c_10 = (2 * j) * 2 + 2
        c_11 = (2 * j) * 2 + 3
        
        # Add Strassen for A1 * B1,j
        add_strassen(a1_00, a1_01, a1_10, a1_11, b1_00, b1_01, b1_10, b1_11, c_00, c_01, c_10, c_11, col)
        col += 7
        
        # Add Strassen for A2 * B2,j
        add_strassen(a2_00, a2_01, a2_10, a2_11, b2_00, b2_01, b2_10, b2_11, c_00, c_01, c_10, c_11, col)
        col += 7

    # Handle j=3 (width 1, col 6)
    # B1,3 indices (rows 0,1; col 6)
    b1_0 = 0 * 7 + 6
    b1_1 = 1 * 7 + 6
    
    # B2,3 indices (rows 2,3; col 6)
    b2_0 = 2 * 7 + 6
    b2_1 = 3 * 7 + 6
    
    # C indices (rows 0,1; col 6)
    c_0 = 6 * 2 + 0
    c_1 = 6 * 2 + 1
    
    # A1 * B1,3 (4 terms)
    U[a1_00][col] += 1
    V[b1_0][col] += 1
    W[c_0][col] += 1
    col += 1
    
    U[a1_01][col] += 1
    V[b1_1][col] += 1
    W[c_0][col] += 1
    col += 1
    
    U[a1_10][col] += 1
    V[b1_0][col] += 1
    W[c_1][col] += 1
    col += 1
    
    U[a1_11][col] += 1
    V[b1_1][col] += 1
    W[c_1][col] += 1
    col += 1
    
    # A2 * B2,3 (4 terms)
    U[a2_00][col] += 1
    V[b2_0][col] += 1
    W[c_0][col] += 1
    col += 1
    
    U[a2_01][col] += 1
    V[b2_1][col] += 1
    W[c_0][col] += 1
    col += 1
    
    U[a2_10][col] += 1
    V[b2_0][col] += 1
    W[c_1][col] += 1
    col += 1
    
    U[a2_11][col] += 1
    V[b2_1][col] += 1
    W[c_1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
