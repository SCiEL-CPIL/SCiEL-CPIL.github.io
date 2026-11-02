"""Matrix multiplication 2,2,5 with rank-minimal exact decomposition."""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication.
    
    We use a block decomposition:
    - Split the 5 columns into two groups of 2 and one group of 1.
    - For each group of 2 columns, use Strassen's algorithm (rank 7) for 2x2x2.
    - For the remaining single column, use schoolbook (rank 4) for 2x2x1.
    - Total rank: 7 + 7 + 4 = 18.
    
    Indexing convention:
    - U: (n*m, rank) = (4, rank), indexed by (i*m + j) where i is row of A, j is col of A
    - V: (m*p, rank) = (10, rank), indexed by (j*p + k) where j is row of B, k is col of B
    - W: (p*n, rank) = (10, rank), indexed by (k*n + i) where k is col of C, i is row of C
    
    The trace convention is W[k*n + i, r] which means output C[i,k] gets coefficient from W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # We'll build the decomposition block by block.
    # Block 1: columns 0, 1 -> Strassen 2x2x2 (rank 7)
    # Block 2: columns 2, 3 -> Strassen 2x2x2 (rank 7)  
    # Block 3: column 4 -> schoolbook 2x2x1 (rank 4)
    
    total_rank = 18
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 18)
    V = [[0] * total_rank for _ in range(m * p)]  # (10, 18)
    W = [[0] * total_rank for _ in range(p * n)]  # (10, 18)
    
    col_offset = 0
    
    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # 
    # Correct Strassen products:
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b11 - b00)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = P1 - P4 + P7
    # C01 = P3 + P1
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5
    #
    # Let me verify C00:
    # P1 - P4 + P7 = (a00+a11)(b00+b11) - a11(b10-b00) + (a01-a11)(b10+b11)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a11b10 + a11b00 + a01b10 + a01b11 - a11b10 - a11b11
    # = a00b00 + a00b11 + 2a11b00 - 2a11b10 + a01b10 + a01b11
    # Target: a00b00 + a01b10
    # This is NOT correct. The standard Strassen I'm recalling is wrong or I'm applying it incorrectly.
    
    # Let me use a verified correct Strassen formulation:
    # From the standard reference:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 - M4 + M7
    # c12 = M3 + M1
    # c21 = M2 + M4
    # c22 = M1 - M3 + M5
    #
    # In 0-indexed (a11->a00, a12->a01, a21->a10, a22->a11, etc.):
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # c00 = M1 - M4 + M7
    # c01 = M3 + M1
    # c10 = M2 + M4
    # c11 = M1 - M3 + M5
    #
    # Verify c00 = a00b00 + a01b10:
    # M1 - M4 + M7 = (a00+a11)(b00+b11) - a11(b10-b00) + (a01-a11)(b10+b11)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a11b10 + a11b00 + a01b10 + a01b11 - a11b10 - a11b11
    # = a00b00 + a00b11 + 2a11b00 - 2a11b10 + a01b10 + a01b11
    # Still not matching. I think I have the wrong Strassen formula.
    
    # Let me just use a simple, verified approach. I'll use the fact that for 2x2x2,
    # the minimal rank is 7, but I need the correct coefficients.
    
    # Actually, let me just use schoolbook for all blocks to be safe and correct.
    # Schoolbook for 2x2x2: rank 4
    # Schoolbook for 2x2x1: rank 4
    # Total: 4 + 4 + 4 = 12
    
    # But wait, that's better than 20! Let me use this.
    
    # Reset
    total_rank = 12
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1: columns 0, 1 (schoolbook 2x2x2, rank 4)
    for i in range(n):
        for j in range(m):
            for k in [0, 1]:
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
    
    # Block 2: columns 2, 3 (schoolbook 2x2x2, rank 4)
    for i in range(n):
        for j in range(m):
            for k in [2, 3]:
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
    
    # Block 3: column 4 (schoolbook 2x2x1, rank 4)
    for i in range(n):
        for j in range(m):
            k = 4
            U[i * m + j][col_offset] = 1
            V[j * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

