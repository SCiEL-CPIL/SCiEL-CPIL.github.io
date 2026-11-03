"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-36 exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 2x2x2 blocks (rank 7 each) for the
    first 4 columns of the result, plus schoolbook (rank 8) for the last column.
    
    For a 2x2 block with A being 2x2, B being 2x2, C being 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 - M4 + M5 + M6
    C01 = M3 + M4
    C10 = M1 + M2 - M5 - M7
    C11 = M2 + M3 + M7
    
    Index conventions:
    U[i*m + j, c] where i is row of A (0,1), j is col of A (0,3)
    V[j*p + k, c] where j is row of B (0,3), k is col of B (0,4)
    W[k*n + i, c] where k is col of C (0,4), i is row of C (0,1)
    """
    n, m, p = 2, 4, 5  # A is n x m, B is m x p, C is n x p
    
    # We'll build the three factor matrices U, V, W as lists of lists.
    # Total rank: 4 blocks * 7 (Strassen) + 8 (schoolbook for last col) = 28 + 8 = 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Add Strassen decomposition for a 2x2 block.
        
        a_row_offset: 0 or 1 (which 2 rows of A)
        a_col_offset: 0 or 2 (which 2 cols of A)
        b_row_offset: 0 or 2 (which 2 rows of B)
        b_col_offset: 0 or 2 (which 2 cols of B)
        c_col_offset: 0 or 2 (which 2 cols of C)
        start_col: starting column in U, V, W
        
        The 7 Strassen terms:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 - M4 + M5 + M6
        C01 = M3 + M4
        C10 = M1 + M2 - M5 - M7
        C11 = M2 + M3 + M7
        """
        # For each of the 7 terms, define the contributions to U, V, W
        
        # Term M1: (A00+A11)(B00+B11)
        # A entries: (a_row_offset+0, a_col_offset+0), (a_row_offset+1, a_col_offset+1)
        # B entries: (b_row_offset+0, b_col_offset+0), (b_row_offset+1, b_col_offset+1)
        # C entries: contributes to C00, C10
        col = start_col + 0
        U[a_row_offset * m + a_col_offset + 0][col] += 1
        U[a_row_offset * m + a_col_offset + 1][col] += 1
        V[b_row_offset * p + b_col_offset + 0][col] += 1
        V[b_row_offset * p + b_col_offset + 1][col] += 1
        W[(c_col_offset + 0) * n + 0][col] += 1   # C00
        W[(c_col_offset + 0) * n + 1][col] += 1   # C10
        
        # Term M2: (A10+A11)B00
        # A entries: (a_row_offset+1, a_col_offset+0), (a_row_offset+1, a_col_offset+1)
        # B entries: (b_row_offset+0, b_col_offset+0)
        # C entries: contributes to C10, C11
        col = start_col + 1
        U[a_row_offset * m + a_col_offset + 0][col] += 1
        U[a_row_offset * m + a_col_offset + 1][col] += 1
        V[b_row_offset * p + b_col_offset + 0][col] += 1
        W[(c_col_offset + 0) * n + 1][col] += 1   # C10
        W[(c_col_offset + 1) * n + 1][col] += 1   # C11
        
        # Term M3: A00(B10-B11)
        # A entries: (a_row_offset+0, a_col_offset+0)
        # B entries: (b_row_offset+1, b_col_offset+0), (b_row_offset+1, b_col_offset+1)
        # C entries: contributes to C01, C11
        col = start_col + 2
        U[a_row_offset * m + a_col_offset + 0][col] += 1
        V[b_row_offset * p + b_col_offset + 0][col] += 1
        V[b_row_offset * p + b_col_offset + 1][col] -= 1
        W[(c_col_offset + 1) * n + 0][col] += 1   # C01
        W[(c_col_offset + 1) * n + 1][col] += 1   # C11
        
        # Term M4: A11(B01-B10)
        # A entries: (a_row_offset+1, a_col_offset+1)
        # B entries: (b_row_offset+0, b_col_offset+1), (b_row_offset+1, b_col_offset+0)
        # C entries: contributes to C00, C01
        col = start_col + 3
        U[a_row_offset * m + a_col_offset + 1][col] += 1
        V[b_row_offset * p + b_col_offset + 1][col] += 1
        V[b_row_offset * p + b_col_offset + 0][col] -= 1
        W[(c_col_offset + 0) * n + 0][col] -= 1   # C00
        W[(c_col_offset + 1) * n + 0][col] += 1   # C01
        
        # Term M5: (A00+A01)B11
        # A entries: (a_row_offset+0, a_col_offset+0), (a_row_offset+0, a_col_offset+1)
        # B entries: (b_row_offset+1, b_col_offset+1)
        # C entries: contributes to C00, C10
        col = start_col + 4
        U[a_row_offset * m + a_col_offset + 0][col] += 1
        U[a_row_offset * m + a_col_offset + 1][col] += 1
        V[b_row_offset * p + b_col_offset + 1][col] += 1
        W[(c_col_offset + 0) * n + 0][col] += 1   # C00
        W[(c_col_offset + 0) * n + 1][col] -= 1   # C10
        
        # Term M6: (A10-A00)(B00+B01)
        # A entries: (a_row_offset+1, a_col_offset+0), (a_row_offset+0, a_col_offset+0)
        # B entries: (b_row_offset+0, b_col_offset+0), (b_row_offset+0, b_col_offset+1)
        # C entries: contributes to C00
        col = start_col + 5
        U[a_row_offset * m + a_col_offset + 0][col] += 1
        U[a_row_offset * m + a_col_offset + 1][col] -= 1
        V[b_row_offset * p + b_col_offset + 0][col] += 1
        V[b_row_offset * p + b_col_offset + 1][col] += 1
        W[(c_col_offset + 0) * n + 0][col] += 1   # C00
        
        # Term M7: (A01-A11)(B10+B11)
        # A entries: (a_row_offset+0, a_col_offset+1), (a_row_offset+1, a_col_offset+1)
        # B entries: (b_row_offset+1, b_col_offset+0), (b_row_offset+1, b_col_offset+1)
        # C entries: contributes to C10, C11
        col = start_col + 6
        U[a_row_offset * m + a_col_offset + 1][col] += 1
        U[a_row_offset * m + a_col_offset + 1][col] -= 1
        V[b_row_offset * p + b_col_offset + 0][col] += 1
        V[b_row_offset * p + b_col_offset + 1][col] += 1
        W[(c_col_offset + 0) * n + 1][col] -= 1   # C10
        W[(c_col_offset + 1) * n + 1][col] += 1   # C11
        
        return start_col + 7  # next free column
    
    # Four 2x2 blocks for the first 4 columns
    # Block (0,0): A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C cols 0-1
    next_col = strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block (0,2): A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C cols 2-3
    next_col = strassen_block(0, 0, 0, 2, 2, next_col)
    
    # Block (2,0): A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C cols 0-1
    next_col = strassen_block(0, 2, 2, 0, 0, next_col)
    
    # Block (2,2): A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C cols 2-3
    next_col = strassen_block(0, 2, 2, 2, 2, next_col)
    
    # Last column (column 4 of C): schoolbook decomposition
    # C[i,4] = sum_j A[i,j] * B[j,4] for i in {0,1}, j in {0,1,2,3}
    for i in range(n):
        for j in range(m):
            col = next_col
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            next_col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
