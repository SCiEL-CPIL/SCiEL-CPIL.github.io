"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition based on Strassen's algorithm for 2x2 matrices.
    # A is 2x4, B is 4x8.
    # Split A into A1, A2 (2x2 each).
    # Split B into B1, B2 (2x8 each, top and bottom).
    # C = A1 B1 + A2 B2.
    # Each term is 2x2 * 2x8.
    # Split 2x8 into four 2x2 blocks.
    # Each 2x2 * 2x2 uses Strassen (rank 7).
    # So each 2x2 * 2x8 uses 4 * 7 = 28 rank.
    # Total rank = 28 + 28 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen 2x2 * 2x2 decomposition
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = XY
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_row_start, b_row_start, c_col_start, offset):
        nonlocal col
        # a is 2x2, b is 2x2, c is 2x2
        # Indices in U, V, W
        # U indices: a_row_start, a_row_start+1 (rows of A)
        # V indices: b_row_start, b_row_start+1 (rows of B)
        # W indices: c_row_start, c_row_start+1 (cols of C)
        
        # M1
        U[a_row_start][col + offset] = 1
        U[a_row_start + 1][col + offset] = 1
        V[b_row_start][col + offset] = 1
        V[b_row_start + 1][col + offset] = 1
        W[c_col_start][col + offset] = 1
        W[c_col_start + 1][col + offset] = 1
        col += 1
        
        # M2
        U[a_row_start + 1][col + offset] = 1
        V[b_row_start][col + offset] = 1
        W[c_col_start + 1][col + offset] = 1
        col += 1
        
        # M3
        U[a_row_start][col + offset] = 1
        V[b_row_start + 1][col + offset] = 1
        W[c_col_start][col + offset] = 1
        col += 1
        
        # M4
        U[a_row_start + 1][col + offset] = 1
        V[b_row_start + 1][col + offset] = 1
        W[c_col_start][col + offset] = 1
        W[c_col_start + 1][col + offset] = 1
        col += 1
        
        # M5
        U[a_row_start][col + offset] = 1
        U[a_row_start + 1][col + offset] = 1
        V[b_row_start + 1][col + offset] = 1
        W[c_col_start + 1][col + offset] = 1
        col += 1
        
        # M6
        U[a_row_start + 1][col + offset] = 1
        U[a_row_start][col + offset] = -1
        V[b_row_start][col + offset] = 1
        V[b_row_start + 1][col + offset] = 1
        W[c_col_start + 1][col + offset] = 1
        col += 1
        
        # M7
        U[a_row_start + 1][col + offset] = 1
        U[a_row_start][col + offset] = -1
        V[b_row_start + 1][col + offset] = 1
        W[c_col_start][col + offset] = 1
        col += 1

    # A is 2x4: Rows 0,1. Cols 0,1,2,3.
    # A1 is cols 0,1. A2 is cols 2,3.
    # B is 4x8: Rows 0,1,2,3. Cols 0..7.
    # B1 is rows 0,1. B2 is rows 2,3.
    
    # Term 1: A1 (2x2) * B1 (2x8)
    # B1 is 2x8, split into 4 blocks of 2x2: B1_0 (cols 0,1), B1_1 (cols 2,3), B1_2 (cols 4,5), B1_3 (cols 6,7)
    for k in range(4):
        # A1 is at A rows 0,1 cols 0,1.
        # B1_k is at B rows 0,1 cols 2k, 2k+1.
        # Result C is at C rows 0,1 cols 2k, 2k+1.
        add_strassen_block(0, 0, 2*k, 0)
        
    # Term 2: A2 (2x2) * B2 (2x8)
    # A2 is at A rows 0,1 cols 2,3.
    # B2 is at B rows 2,3.
    for k in range(4):
        # A2 is at A rows 0,1 cols 2,3.
        # B2_k is at B rows 2,3 cols 2k, 2k+1.
        # Result C is at C rows 0,1 cols 2k, 2k+1.
        add_strassen_block(2, 2, 2*k, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
