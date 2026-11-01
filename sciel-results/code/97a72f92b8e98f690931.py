"""Program for matrix multiplication 2x4 * 4x7; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # We use a hybrid approach: split the 4x7 product into 2x(4x7) blocks.
    # Specifically, split the 4 rows of B (and 4 columns of A) into two groups of 2.
    # Group 1: rows 0,1 of B (cols 0,1 of A) -> contributes to A[0:2, 0:2] * B[0:2, :]
    # Group 2: rows 2,3 of B (cols 2,3 of A) -> contributes to A[0:2, 2:4] * B[2:4, :]
    
    # For each 2x2 * 2x7 block, we can use a decomposition.
    # Let's just use the schoolbook for each 2x2*2x7 block? That would be 2*(2*2*7) = 56. No gain.
    # We need a better decomposition for 2x2 * 2x7.
    
    # Let's try to find a low-rank decomposition for 2x2 * 2x7.
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k} where i in 0..1, j in 0..1, k in 0..6.
    # Rank is at most 2*2*7 = 28.
    # Can we do better?
    # For 2x2 * 2x2, rank is 7 (Strassen).
    # For 2x2 * 2x7, we can view it as 2x2 * (2x2 + 2x2 + 2x1).
    # Block 1: 2x2 * 2x2 (cols 0,1 of k) -> rank 7
    # Block 2: 2x2 * 2x2 (cols 2,3 of k) -> rank 7
    # Block 3: 2x2 * 2x2 (cols 4,5 of k) -> rank 7
    # Block 4: 2x2 * 2x1 (col 6 of k) -> rank 4 (schoolbook: 2*2*1=4)
    # Total rank = 7 + 7 + 7 + 4 = 25.
    
    # Let's verify if 2x2 * 2x1 can be done in rank 4.
    # A = [[a,b],[c,d]], B = [[e],[f]] (2x1)
    # C = [[ae+bf],[ce+df]] (2x1)
    # C_0 = ae + bf
    # C_1 = ce + df
    # This is just 4 bilinear terms. Rank 4.
    
    # So total rank for the 2x2 * 2x7 block is 25.
    # We have two such blocks (for the two 2x2 sub-blocks of A/B).
    # Total rank = 25 + 25 = 50.
    # This is better than 56.
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 50
    
    # Helper to add a rank-1 term
    def add_term(U_idx, V_idx, W_idx, coeff=1.0):
        # This is not how it works. U, V, W are matrices where each column is a rank-1 term.
        # We need to define the rank-1 terms directly.
        pass
    
    # We will construct the terms explicitly.
    
    # Block 1: A[0:2, 0:2] * B[0:2, :]
    # A indices: 0,1 (i=0,j=0; i=0,j=1) and 4,5 (i=1,j=0; i=1,j=1)
    # B indices: 0..6 (j=0,k=0..6) and 7..13 (j=1,k=0..6)
    # C indices: 0,2 (i=0,k=0; i=1,k=0), 1,3 (i=0,k=1; i=1,k=1), ..., 12,14 (i=0,k=6; i=1,k=6)
    
    # Let's define a function to add a Strassen decomposition for a 2x2 * 2x2 block.
    # Inputs: A_00, A_01, A_10, A_11, B_00, B_01, B_10, B_11
    # Outputs: C_00, C_01, C_10, C_11
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B10)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        # M3
        U[a00][col] += 1
        V[b11][col] += 1; V[b10][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += -1; W[c10][col] += 1
        col += 1
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        # M6
        U[a10][col] += 1; U[a00][col] += -1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        # M7
        U[a01][col] += 1; U[a11][col] += -1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += -1
        col += 1

    # Block 4: 2x2 * 2x1
    def add_2x2_x_2x1(a00, a01, a10, a11, b00, b10, c0, c1):
        nonlocal col
        # C0 = a00*b00 + a01*b10
        # C1 = a10*b00 + a11*b10
        # Term 1: a00*b00
        U[a00][col] += 1
        V[b00][col] += 1
        W[c0][col] += 1
        col += 1
        # Term 2: a01*b10
        U[a01][col] += 1
        V[b10][col] += 1
        W[c0][col] += 1
        col += 1
        # Term 3: a10*b00
        U[a10][col] += 1
        V[b00][col] += 1
        W[c1][col] += 1
        col += 1
        # Term 4: a11*b10
        U[a11][col] += 1
        V[b10][col] += 1
        W[c1][col] += 1
        col += 1

    # Process Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # A indices: 0,1,4,5
    # B indices: 0..6, 7..13
    # C indices: 0,2, 1,3, 4,6, 5,7, 8,10, 9,11, 12,14
    
    # Sub-block 1.1: k=0,1 (B cols 0,1 and 7,8)
    # B_00=0, B_01=1, B_10=7, B_11=8
    # C_00=0, C_01=1, C_10=2, C_11=3
    add_strassen(0, 1, 4, 5, 0, 1, 7, 8, 0, 1, 2, 3)
    
    # Sub-block 1.2: k=2,3 (B cols 2,3 and 9,10)
    # B_00=2, B_01=3, B_10=9, B_11=10
    # C_00=4, C_01=5, C_10=6, C_11=7
    add_strassen(0, 1, 4, 5, 2, 3, 9, 10, 4, 5, 6, 7)
    
    # Sub-block 1.3: k=4,5 (B cols 4,5 and 11,12)
    # B_00=4, B_01=5, B_10=11, B_11=12
    # C_00=8, C_01=9, C_10=10, C_11=11
    add_strassen(0, 1, 4, 5, 4, 5, 11, 12, 8, 9, 10, 11)
    
    # Sub-block 1.4: k=6 (B col 6 and 13)
    # B_00=6, B_10=13
    # C_0=12, C_1=13
    add_2x2_x_2x1(0, 1, 4, 5, 6, 13, 12, 13)
    
    # Process Block 2: A[0:2, 2:4] * B[2:4, 0:7]
    # A indices: 2,3 (i=0,j=2; i=0,j=3) and 6,7 (i=1,j=2; i=1,j=3)
    # B indices: 14..20 (j=2,k=0..6) and 21..27 (j=3,k=0..6)
    # C indices: same as before 0..13
    
    # Sub-block 2.1: k=0,1
    # B_00=14, B_01=15, B_10=21, B_11=22
    # C_00=0, C_01=1, C_10=2, C_11=3
    add_strassen(2, 3, 6, 7, 14, 15, 21, 22, 0, 1, 2, 3)
    
    # Sub-block 2.2: k=2,3
    # B_00=16, B_01=17, B_10=23, B_11=24
    # C_00=4, C_01=5, C_10=6, C_11=7
    add_strassen(2, 3, 6, 7, 16, 17, 23, 24, 4, 5, 6, 7)
    
    # Sub-block 2.3: k=4,5
    # B_00=18, B_01=19, B_10=25, B_11=26
    # C_00=8, C_01=9, C_10=10, C_11=11
    add_strassen(2, 3, 6, 7, 18, 19, 25, 26, 8, 9, 10, 11)
    
    # Sub-block 2.4: k=6
    # B_00=20, B_10=27
    # C_0=12, C_1=13
    add_2x2_x_2x1(2, 3, 6, 7, 20, 27, 12, 13)
    
    # Check col
    assert col == rank, f"col {col} != rank {rank}"
    
    # Convert to list of lists (already is)
    # Ensure entries are integers or half-integers (they are integers here)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
