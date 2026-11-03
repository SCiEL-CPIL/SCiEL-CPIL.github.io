import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-88 decomposition for 3x4x8 matrix multiplication.
    
    Strategy:
    - Split the 4x8 product into a 2x2 block (top-left) and remaining parts.
    - The 2x2x2 sub-product uses Strassen's algorithm (rank 7).
    - The 2x2x2 block is replicated 4 times to cover 2x2x8 (rank 7*4 = 28).
    - The remaining parts (first row of 4x8, bottom 2x4, bottom 2x4) are handled
      by schoolbook (rank 24 + 16 + 16 = 56).
    - Total rank: 28 + 56 = 84? Wait, let me recompute.
    
    Actually, let me use the known rank-88 construction:
    - Partition A (3x4) into A11 (2x2), A12 (2x2), A21 (1x2), A22 (1x2)
    - Partition B (4x8) into B11 (2x8), B21 (2x8), B12 (2x8), B22 (2x8)
      Wait, B is 4x8, so we split rows: first 2 rows, last 2 rows.
      B = [B_top (2x8); B_bottom (2x8)]
    
    C = A * B = [A11 A12] * [B_top; B_bottom] + [A21 A22] * [B_top; B_bottom]
              = A11 * B_top + A12 * B_bottom + A21 * B_top + A22 * B_bottom
    
    Where:
    - A11: 2x2, B_top: 2x8 -> product 2x8, rank 7*4 = 28 (Strassen on 2x2x2 blocks)
    - A12: 2x2, B_bottom: 2x8 -> product 2x8, rank 7*4 = 28
    - A21: 1x2, B_top: 2x8 -> product 1x8, rank 8 (schoolbook: 1*2*8/2 = 8? No, 2*8=16)
    - A22: 1x2, B_bottom: 2x8 -> product 1x8, rank 16
    
    Total: 28 + 28 + 16 + 16 = 88. Yes!
    
    For the 2x2x8 case using Strassen:
    We have A2x2 and B2x8. B can be split into 4 blocks of 2x2: B = [B1, B2, B3, B4]
    Each 2x2x2 multiplication uses Strassen (rank 7).
    So 2x2x8 has rank 7*4 = 28.
    
    For the 1x2x8 case (A is 1x2, B is 2x8):
    C[i,k] = sum_j A[j] * B[j,k]
    This is just a weighted sum of two rows of B.
    Schoolbook: for each of the 8 columns, we need 2 multiplications.
    So rank = 1 * 2 * 8 = 16.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # For matrices A (2x2), B (2x2), C = A*B (2x2)
    # Strassen: 7 products
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(u_row1, u_row2, v_row1, v_row2, w_row1, w_row2, w_row3, w_row4, start_col):
        """
        Add 7 Strassen terms for 2x2x2 multiplication.
        u_row1, u_row2: rows of U for the 2x2 A block (2 entries each in U)
        v_row1, v_row2: rows of V for the 2x2 B block
        w_row1..w_row4: rows of W for the 2x2 C block
        
        Actually, let me be more explicit.
        For a 2x2x2 multiplication at position (i_offset, j_offset, k_offset):
        A is 2x2: A[i1, j1], A[i1, j2], A[i2, j1], A[i2, j2]
        B is 2x2: B[j1, k1], B[j1, k2], B[j2, k1], B[j2, k2]
        C is 2x2: C[k1, i1], C[k1, i2], C[k2, i1], C[k2, i2]
        
        U indices: (i1*m + j1), (i1*m + j2), (i2*m + j1), (i2*m + j2)
        V indices: (j1*p + k1), (j1*p + k2), (j2*p + k1), (j2*p + k2)
        W indices: (k1*n + i1), (k1*n + i2), (k2*n + i1), (k2*n + i2)
        """
        # M1 = (A11 + A22)(B11 + B22)
        c = start_col
        U[u_row1][c] = 1; V[v_row1][c] = 1; W[w_row1][c] = 1
        U[u_row2][c] = 1; V[v_row2][c] = 1; W[w_row2][c] = 1
        c += 1
        # M2 = (A21 + A22)B11
        U[u_row2][c] = 1; V[v_row1][c] = 1; W[w_row3][c] = 1
        U[u_row1][c] = 1; W[w_row3][c] = 1
        c += 1
        # M3 = A11(B12 - B22)
        U[u_row1][c] = 1; V[v_row2][c] = 1; W[w_row2][c] = 1
        V[v_row3][c] = -1; W[w_row1][c] = 1
        c += 1
        # M4 = A22(B21 - B11)
        U[u_row2][c] = 1; V[v_row3][c] = 1; W[w_row3][c] = 1
        V[v_row1][c] = -1
        c += 1
        # M5 = (A11 + A12)B22
        U[u_row1][c] = 1; V[v_row3][c] = 1; W[w_row2][c] = 1
        U[u_row2][c] = 1
        c += 1
        # M6 = (A21 - A11)(B11 + B12)
        U[u_row2][c] = 1; V[v_row1][c] = 1; W[w_row4][c] = 1
        U[u_row1][c] = -1; V[v_row2][c] = 1
        c += 1
        # M7 = (A12 - A22)(B21 + B22)
        U[u_row2][c] = 1; V[v_row3][c] = 1; W[w_row1][c] = 1
        U[u_row1][c] = -1; V[v_row2][c] = 1
        c += 1
        return c
    
    # 2x2x8 using Strassen: A is 2x2, B is 2x8
    # B is split into 4 blocks of 2x2
    # A rows: i1, i2; B rows: j1, j2
    # B columns: k1..k8, split into (k1,k2), (k3,k4), (k5,k6), (k7,k8)
    
    def strassen_2x2x8(i1, i2, j1, j2, k_start, col_offset):
        """
        i1, i2: row indices in A (0-indexed, offset by n*m)
        j1, j2: row indices in B (0-indexed, offset by m*p)
        k_start: starting column index in B (0-indexed)
        col_offset: starting column in the factor matrices
        """
        # u indices for A
        u_i1j1 = i1 * m + j1
        u_i1j2 = i1 * m + j2
        u_i2j1 = i2 * m + j1
        u_i2j2 = i2 * m + j2
        
        for block in range(4):
            k1 = k_start + block * 2
            k2 = k_start + block * 2 + 1
            # v indices for B block
            v_j1k1 = j1 * p + k1
            v_j1k2 = j1 * p + k2
            v_j2k1 = j2 * p + k1
            v_j2k2 = j2 * p + k2
            # w indices for C block
            w_k1i1 = k1 * n + i1
            w_k1i2 = k1 * n + i2
            w_k2i1 = k2 * n + i1
            w_k2i2 = k2 * n + i2
            col_offset = strassen_2x2x2(
                u_i1j1, u_i2j2,  # These are just row identifiers, not the actual values
                v_j1k1, v_j2k2,
                w_k1i1, w_k1i2, w_k2i1, w_k2i2,
                col_offset
            )
        return col_offset
    
    # 1x2x8 schoolbook: A is 1x2, B is 2x8
    def schoolbook_1x2x8(i1, j1, j2, k_start, col_offset):
        """
        i1: row index in A
        j1, j2: row indices in B
        k_start: starting column index in B
        """
        u_i1j1 = i1 * m + j1
        u_i1j2 = i1 * m + j2
        for k in range(p):
            c = col_offset
            U[u_i1j1][c] = 1
            V[j1 * p + k][c] = 1
            W[k * n + i1][c] = 1
            col_offset += 1
            U[u_i1j2][c] = 1
            V[j2 * p + k][c] = 1
            W[k * n + i1][c] = 1
            col_offset += 1
        return col_offset
    
    # Partition:
    # A (3x4): rows 0,1,2; cols 0,1,2,3
    # A11 = A[0:2, 0:2] -> 2x2
    # A12 = A[0:2, 2:4] -> 2x2
    # A21 = A[2:3, 0:2] -> 1x2
    # A22 = A[2:3, 2:4] -> 1x2
    #
    # B (4x8): rows 0,1,2,3; cols 0..7
    # B_top = B[0:2, :] -> 2x8
    # B_bottom = B[2:4, :] -> 2x8
    #
    # C = A11 * B_top + A12 * B_bottom + A21 * B_top + A22 * B_bottom
    #
    # C1 = A11 * B_top: 2x8, rows 0,1 of C
    # C2 = A12 * B_bottom: 2x8, rows 0,1 of C
    # C3 = A21 * B_top: 1x8, row 2 of C
    # C4 = A22 * B_bottom: 1x8, row 2 of C
    
    # C1: A11 (2x2) * B_top (2x8)
    # A11 rows: 0,1; cols: 0,1
    # B_top rows: 0,1; cols: 0..7
    col = strassen_2x2x8(0, 1, 0, 1, 0, col)  # 28 terms
    
    # C2: A12 (2x2) * B_bottom (2x8)
    # A12 rows: 0,1; cols: 2,3
    # B_bottom rows: 2,3; cols: 0..7
    col = strassen_2x2x8(0, 1, 2, 3, 0, col)  # 28 terms
    
    # C3: A21 (1x2) * B_top (2x8)
    # A21 row: 2; cols: 0,1
    # B_top rows: 0,1; cols: 0..7
    col = schoolbook_1x2x8(2, 0, 1, 0, col)  # 16 terms
    
    # C4: A22 (1x2) * B_bottom (2x8)
    # A22 row: 2; cols: 2,3
    # B_bottom rows: 2,3; cols: 0..7
    col = schoolbook_1x2x8(2, 2, 3, 0, col)  # 16 terms
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
