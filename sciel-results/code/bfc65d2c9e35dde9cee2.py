"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Construct a rank-8 decomposition for 2x3 by 3x4 matrix multiplication.
    # This is achieved by splitting the 3x3 intermediate dimension into 2x2 and 1x1 parts.
    # Part 1: 2x2 by 2x4 multiplication (using two 2x2x2 Strassen blocks, rank 7+7=14? No, we optimized)
    # Wait, my previous derivation:
    # A (2x3) = [A1 (2x2), A2 (2x1)]
    # B (3x4) = [B1 (2x4); B2 (1x4)]
    # C = A1 B1 + A2 B2
    # A1 B1 is 2x2 by 2x4.
    # Split B1 into B1a (2x2) and B1b (2x2).
    # A1 B1a is 2x2x2 (Rank 7).
    # A1 B1b is 2x2x2 (Rank 7).
    # So A1 B1 has rank 14.
    # A2 B2 is 2x1 by 1x4 (Rank 1).
    # Total rank 15.
    
    # Let's re-verify the rank of M_{2,2,4}.
    # A (2x2), B (2x4).
    # B = [B1, B2] where B1, B2 are 2x2.
    # C = A B1 + A B2.
    # Rank(A B1) <= 7. Rank(A B2) <= 7.
    # Total <= 14.
    # Is it possible to do better?
    # The tensor rank of M_{2,2,4} is known to be 14?
    # Actually, M_{2,2,4} is isomorphic to M_{2,4,2}^T? No.
    # Let's stick with 14 + 1 = 15.
    
    # Wait, can we share factors between A1 B1a and A1 B1b?
    # No, the B inputs are different.
    # So rank 15 is a safe upper bound.
    # Score 24/15 = 1.6.
    
    # Let's try to optimize M_{2,2,4} further?
    # No, 14 is likely minimal for that block.
    
    # Let's implement rank 15.
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    rank = 15
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 15
    V = [[0] * rank for _ in range(m * p)] # 12 x 15
    W = [[0] * rank for _ in range(p * n)] # 8 x 15
    
    # Part 1: A1 (2x2) * B1 (2x4)
    # B1 is split into B1a (2x2) and B1b (2x2)
    # B1a corresponds to B cols 0,1 (k=0,1)
    # B1b corresponds to B cols 2,3 (k=2,3)
    
    # Strassen 1 for B1a
    # U1a (4x7), V1a (4x7), W1a (4x7)
    # Strassen 2 for B1b
    # U1b (4x7), V1b (4x7), W1b (4x7)
    
    # Part 2: A2 (2x1) * B2 (1x4)
    # U2 (2x1), V2 (4x1), W2 (4x2)
    
    # Helper to add Strassen block
    def add_strassen(u_offset, v_offset, w_offset, col_offset):
        # u_offset: row index in U for A1 (i=0,1; j=0,1) -> global i, j
        # v_offset: row index in V for B1 (j=0,1; k=0,1 or 2,3) -> global j, k
        # w_offset: row index in W for C (k=0,1; i=0,1) -> global k, i
        
        # Strassen coefficients
        # U: A11, A12, A21, A22
        # V: B11, B12, B21, B22
        # W: C11, C12, C21, C22
        
        # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # U values
        U[u_offset + 0][col_offset + 0] = 1  # A11 M1
        U[u_offset + 0][col_offset + 2] = 1  # A11 M3
        U[u_offset + 0][col_offset + 4] = 1  # A11 M5
        U[u_offset + 0][col_offset + 5] = -1 # A11 M6
        U[u_offset + 0][col_offset + 6] = 1  # A11 M7
        
        U[u_offset + 1][col_offset + 4] = 1  # A12 M5
        U[u_offset + 1][col_offset + 6] = -1 # A12 M7
        
        U[u_offset + 2][col_offset + 1] = 1  # A21 M2
        U[u_offset + 2][col_offset + 5] = 1  # A21 M6
        
        U[u_offset + 3][col_offset + 0] = 1  # A22 M1
        U[u_offset + 3][col_offset + 1] = 1  # A22 M2
        U[u_offset + 3][col_offset + 3] = 1  # A22 M4
        U[u_offset + 3][col_offset + 6] = -1 # A22 M7
        
        # V values
        V[v_offset + 0][col_offset + 0] = 1  # B11 M1
        V[v_offset + 0][col_offset + 1] = 1  # B11 M2
        V[v_offset + 0][col_offset + 3] = -1 # B11 M4
        V[v_offset + 0][col_offset + 5] = 1  # B11 M6
        
        V[v_offset + 1][col_offset + 2] = 1  # B12 M3
        V[v_offset + 1][col_offset + 5] = 1  # B12 M6
        
        V[v_offset + 2][col_offset + 3] = 1  # B21 M4
        V[v_offset + 2][col_offset + 6] = 1  # B21 M7
        
        V[v_offset + 3][col_offset + 0] = 1  # B22 M1
        V[v_offset + 3][col_offset + 2] = -1 # B22 M3
        V[v_offset + 3][col_offset + 4] = 1  # B22 M5
        V[v_offset + 3][col_offset + 6] = 1  # B22 M7
        
        # W values
        # C11 = M1 + M4 - M5 + M7
        W[w_offset + 0][col_offset + 0] = 1
        W[w_offset + 0][col_offset + 3] = 1
        W[w_offset + 0][col_offset + 4] = -1
        W[w_offset + 0][col_offset + 6] = 1
        
        # C12 = M3 + M5
        W[w_offset + 1][col_offset + 2] = 1
        W[w_offset + 1][col_offset + 4] = 1
        
        # C21 = M2 + M4
        W[w_offset + 2][col_offset + 1] = 1
        W[w_offset + 2][col_offset + 3] = 1
        
        # C22 = M1 - M2 + M3 + M6
        W[w_offset + 3][col_offset + 0] = 1
        W[w_offset + 3][col_offset + 1] = -1
        W[w_offset + 3][col_offset + 2] = 1
        W[w_offset + 3][col_offset + 5] = 1

    # Map global indices
    # U rows: i*m + j. i in 0,1. j in 0,1,2.
    # A1 uses j=0,1.
    # u_offset for A1 is 0.
    # V rows: j*p + k. j in 0,1,2. k in 0,1,2,3.
    # B1 uses j=0,1.
    # B1a uses k=0,1. v_offset = 0.
    # B1b uses k=2,3. v_offset = 2*p = 8? No.
    # V index = j*p + k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    # So B1a V rows are 0,1,4,5.
    # B1b V rows are 2,3,6,7.
    
    # W rows: k*n + i. k in 0,1,2,3. i in 0,1.
    # C1a uses k=0,1.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # So B1a W rows are 0,1,2,3.
    # B1b uses k=2,3.
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    # So B1b W rows are 4,5,6,7.
    
    # Add Strassen 1 for B1a
    # U offset 0
    # V offset 0 (but rows are 0,1,4,5)
    # My add_strassen assumes contiguous V rows.
    # I need to handle non-contiguous V rows.
    
    # Let's modify add_strassen to take explicit row indices.
    
    def add_strassen_explicit(u_rows, v_rows, w_rows, col_offset):
        # u_rows: [u0, u1, u2, u3] corresponding to A11, A12, A21, A22
        # v_rows: [v0, v1, v2, v3] corresponding to B11, B12, B21, B22
        # w_rows: [w0, w1, w2, w3] corresponding to C11, C12, C21, C22
        
        # U
        U[u_rows[0]][col_offset + 0] = 1
        U[u_rows[0]][col_offset + 2] = 1
        U[u_rows[0]][col_offset + 4] = 1
        U[u_rows[0]][col_offset + 5] = -1
        U[u_rows[0]][col_offset + 6] = 1
        
        U[u_rows[1]][col_offset + 4] = 1
        U[u_rows[1]][col_offset + 6] = -1
        
        U[u_rows[2]][col_offset + 1] = 1
        U[u_rows[2]][col_offset + 5] = 1
        
        U[u_rows[3]][col_offset + 0] = 1
        U[u_rows[3]][col_offset + 1] = 1
        U[u_rows[3]][col_offset + 3] = 1
        U[u_rows[3]][col_offset + 6] = -1
        
        # V
        V[v_rows[0]][col_offset + 0] = 1
        V[v_rows[0]][col_offset + 1] = 1
        V[v_rows[0]][col_offset + 3] = -1
        V[v_rows[0]][col_offset + 5] = 1
        
        V[v_rows[1]][col_offset + 2] = 1
        V[v_rows[1]][col_offset + 5] = 1
        
        V[v_rows[2]][col_offset + 3] = 1
        V[v_rows[2]][col_offset + 6] = 1
        
        V[v_rows[3]][col_offset + 0] = 1
        V[v_rows[3]][col_offset + 2] = -1
        V[v_rows[3]][col_offset + 4] = 1
        V[v_rows[3]][col_offset + 6] = 1
        
        # W
        W[w_rows[0]][col_offset + 0] = 1
        W[w_rows[0]][col_offset + 3] = 1
        W[w_rows[0]][col_offset + 4] = -1
        W[w_rows[0]][col_offset + 6] = 1
        
        W[w_rows[1]][col_offset + 2] = 1
        W[w_rows[1]][col_offset + 4] = 1
        
        W[w_rows[2]][col_offset + 1] = 1
        W[w_rows[2]][col_offset + 3] = 1
        
        W[w_rows[3]][col_offset + 0] = 1
        W[w_rows[3]][col_offset + 1] = -1
        W[w_rows[3]][col_offset + 2] = 1
        W[w_rows[3]][col_offset + 5] = 1

    # Part 1a: A1 * B1a (cols 0,1)
    # U rows: A11(0,0)->0, A12(0,1)->1, A21(1,0)->3, A22(1,1)->4
    u_rows_1a = [0, 1, 3, 4]
    # V rows: B11(0,0)->0, B12(0,1)->1, B21(1,0)->4, B22(1,1)->5
    v_rows_1a = [0, 1, 4, 5]
    # W rows: C11(0,0)->0, C12(0,1)->1, C21(1,0)->2, C22(1,1)->3
    w_rows_1a = [0, 1, 2, 3]
    
    add_strassen_explicit(u_rows_1a, v_rows_1a, w_rows_1a, 0)
    
    # Part 1b: A1 * B1b (cols 2,3)
    # U rows: same as 1a
    u_rows_1b = [0, 1, 3, 4]
    # V rows: B11(0,2)->2, B12(0,3)->3, B21(1,2)->6, B22(1,3)->7
    v_rows_1b = [2, 3, 6, 7]
    # W rows: C11(2,0)->4, C12(2,1)->5, C21(3,0)->6, C22(3,1)->7
    w_rows_1b = [4, 5, 6, 7]
    
    add_strassen_explicit(u_rows_1b, v_rows_1b, w_rows_1b, 7)
    
    # Part 2: A2 * B2
    # A2 is 2x1 (j=2). U rows: (0,2)->2, (1,2)->5
    # B2 is 1x4 (j=2). V rows: (2,0)->8, (2,1)->9, (2,2)->10, (2,3)->11
    # C is 2x4. W rows: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3, (2,0)->4, (2,1)->5, (3,0)->6, (3,1)->7
    
    # Rank 1 term
    col = 14
    U[2][col] = 1
    U[5][col] = 1
    V[8][col] = 1
    V[9][col] = 1
    V[10][col] = 1
    V[11][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    W[2][col] = 1
    W[3][col] = 1
    W[4][col] = 1
    W[5][col] = 1
    W[6][col] = 1
    W[7][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
