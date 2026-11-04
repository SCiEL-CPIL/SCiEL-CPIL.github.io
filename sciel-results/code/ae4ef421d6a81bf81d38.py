"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Matrix multiplication: A (4x4) * B (4x5) = C (4x5)
    # We use a block-based decomposition with Strassen for 2x2x2 blocks
    # and schoolbook for the remaining 1-column blocks.
    #
    # Strategy:
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 blocks horizontally (since B is 4x5, we split rows into 2+2, and columns into 2+2+1)
    # Actually, B is 4x5. Let's think of it as 4 rows, 5 columns.
    # We can split B's rows into 2+2 and columns into 2+2+1.
    #
    # Let's use the following approach:
    # C = A * B
    # Split A into 2x2 blocks (2x2 each), so A is 4x4 -> 4 blocks of 2x2
    # Split B into blocks: B is 4x5. Split rows into 2+2, columns into 2+2+1.
    # So B has blocks:
    #   B11 (2x2), B12 (2x2), B13 (2x1)
    #   B21 (2x2), B22 (2x2), B23 (2x1)
    #
    # Then C = A * B can be computed as:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # For each 2x2 * 2x2 product, we use Strassen (rank 7)
    # For each 2x2 * 2x1 product, we use schoolbook (rank 4)
    #
    # Number of 2x2*2x2 products: C11, C12, C21, C22 -> 4 blocks, each 2 terms -> 8 Strassen computations
    # Number of 2x2*2x1 products: C13, C23 -> 2 blocks, each 2 terms -> 4 schoolbook computations
    #
    # Total rank: 8*7 + 4*4 = 56 + 16 = 72

    n, m, p = 4, 4, 5
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 20 x 72
    W = [[0] * rank for _ in range(p * n)]  # 20 x 72
    
    col = 0
    
    # Helper to add a Strassen term for 2x2*2x2 at specific block positions
    # Strassen formulas for 2x2*2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # Output:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    def add_strassen(ai, aj, bi, bj, ci, cj, start_col):
        """Add 7 Strassen terms for A[ai:ai+2, aj:aj+2] * B[bi:bi+2, bj:bj+2] contributing to C[ci:ci+2, cj:cj+2]"""
        global col
        # A indices: rows ai, ai+1; cols aj, aj+1
        # B indices: rows bi, bi+1; cols bj, bj+1
        # C indices: rows ci, ci+1; cols cj, cj+1
        
        a11 = ai * m + aj      # 0
        a12 = ai * m + aj + 1  # 1
        a21 = (ai + 1) * m + aj      # m
        a22 = (ai + 1) * m + aj + 1  # m+1
        
        b11 = bi * p + bj      # 0
        b12 = bi * p + bj + 1  # 1
        b21 = (bi + 1) * p + bj      # p
        b22 = (bi + 1) * p + bj + 1  # p+1
        
        c11 = cj * n + ci      # 0
        c12 = cj * n + ci + 1  # 1
        c21 = (cj + 1) * n + ci      # n
        c22 = (cj + 1) * n + ci + 1  # n+1
        
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        W[c22][col] = -1
        col += 1
        
        # M3 = a11(b12-b22)
        U[a11][col] = 1
        V[b12][col] = 1
        V[b22][col] = -1
        W[c12][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M4 = a22(b21-b11)
        U[a22][col] = 1
        V[b21][col] = 1
        V[b11][col] = -1
        W[c11][col] = 1
        W[c21][col] = 1
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1
        W[c12][col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x1(ai, aj, bi, bj, ci, cj, start_col):
        """Add 4 schoolbook terms for A[ai:ai+2, aj:aj+2] * B[bi:bi+2, bj:bj+1] contributing to C[ci:ci+2, cj:cj+1]"""
        global col
        a11 = ai * m + aj
        a12 = ai * m + aj + 1
        a21 = (ai + 1) * m + aj
        a22 = (ai + 1) * m + aj + 1
        
        b11 = bi * p + bj
        b21 = (bi + 1) * p + bj
        
        c11 = cj * n + ci
        c21 = (cj + 1) * n + ci
        
        # c11 = a11*b11 + a12*b21
        U[a11][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        col += 1
        
        U[a12][col] = 1
        V[b21][col] = 1
        W[c11][col] = 1
        col += 1
        
        # c21 = a21*b11 + a22*b21
        U[a21][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        col += 1
        
        U[a22][col] = 1
        V[b21][col] = 1
        W[c21][col] = 1
        col += 1
        
        return col
    
    # Block positions:
    # A blocks: (0,0), (0,1), (1,0), (1,1) -> ai in {0,1}, aj in {0,1}
    # B blocks: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> bi in {0,1}, bj in {0,1,2}
    # C blocks: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> ci in {0,1}, cj in {0,1,2}
    
    # C11 = A11*B11 + A12*B21 (both 2x2*2x2)
    col = add_strassen(0, 0, 0, 0, 0, 0, col)   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen(0, 1, 1, 0, 0, 0, col)   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    
    # C12 = A11*B12 + A12*B22 (both 2x2*2x2)
    col = add_strassen(0, 0, 0, 1, 0, 1, col)   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = add_strassen(0, 1, 1, 1, 0, 1, col)   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # C21 = A21*B11 + A22*B21 (both 2x2*2x2)
    col = add_strassen(1, 0, 0, 0, 1, 0, col)   # A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
    col = add_strassen(1, 1, 1, 0, 1, 0, col)   # A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    
    # C22 = A21*B12 + A22*B22 (both 2x2*2x2)
    col = add_strassen(1, 0, 0, 1, 1, 1, col)   # A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4]
    col = add_strassen(1, 1, 1, 1, 1, 1, col)   # A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
    
    # C13 = A11*B13 + A12*B23 (both 2x2*2x1)
    col = add_schoolbook_2x2_2x1(0, 0, 0, 2, 0, 2, col)   # A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    col = add_schoolbook_2x2_2x1(0, 1, 1, 2, 0, 2, col)   # A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    
    # C23 = A21*B13 + A22*B23 (both 2x2*2x1)
    col = add_schoolbook_2x2_2x1(1, 0, 0, 2, 1, 2, col)   # A[2:4,0:2] * B[0:2,4:5] -> C[2:4,4:5]
    col = add_schoolbook_2x2_2x1(1, 1, 1, 2, 1, 2, col)   # A[2:4,2:4] * B[2:4,4:5] -> C[2:4,4:5]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
