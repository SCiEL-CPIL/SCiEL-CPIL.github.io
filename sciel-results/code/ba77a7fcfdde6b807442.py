"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses recursive Strassen:
- 4x4 x 4x4: 7 Strassen multiplications of 2x2 x 2x2 (rank 7 each) -> rank 49
- 4x4 x 4x2: 4 Strassen multiplications of 2x2 x 2x2 (rank 7 each) -> rank 28
- Total rank = 77

Index conventions:
  U[i*m + j, r] for A[i,j]
  V[j*p + k, r] for B[j,k]
  W[k*n + i, r] for C[i,k]  (trace convention)
"""

import time


def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2*2x2 as (u_mask, v_mask, w_mask) tuples.
    Each mask is a dict mapping position to coefficient.
    Positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    terms = []
    # T1: (A00+A11)(B00+B11) -> C00+C01+C10+C11
    terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:1, 3:1}))
    # T2: (A10+A11)(B00) -> C10-C11
    terms.append(({2:1, 3:1}, {0:1}, {2:1, 3:-1}))
    # T3: A00(B10-B11) -> C01+C11
    terms.append(({0:1}, {2:1, 3:-1}, {1:1, 3:1}))
    # T4: A11(B01-B00) -> C00-C11
    terms.append(({3:1}, {1:1, 0:-1}, {0:1, 3:-1}))
    # T5: (A00+A01)(B11) -> -C00+C01
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
    # T6: (A10-A00)(B00+B01) -> C11
    terms.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # T7: (A01-A11)(B10+B11) -> C00
    terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
    return terms


def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    U: shape (16, rank), V: shape (24, rank), W: shape (24, rank)
    """
    start_time = time.time()
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4 x 4x4 (B cols 0-3, C cols 0-3): rank 49
    # Part 2: 4x4 x 4x2 (B cols 4-5, C cols 4-5): rank 28
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    strassen = _strassen_2x2_terms()
    
    def apply_strassen(a_rows, b_cols, c_rows, col):
        """
        Apply Strassen 2x2*2x2 decomposition.
        a_rows: list of 4 U row indices for A[0,0],A[0,1],A[1,0],A[1,1]
        b_cols: list of 4 V col indices for B[0,0],B[0,1],B[1,0],B[1,1]
        c_rows: list of 4 W row indices for C[0,0],C[0,1],C[1,0],C[1,1]
        Returns new col.
        """
        for u_mask, v_mask, w_mask in strassen:
            for pos, coeff in u_mask.items():
                U[a_rows[pos]][col] += coeff
            for pos, coeff in v_mask.items():
                V[b_cols[pos]][col] += coeff
            for pos, coeff in w_mask.items():
                W[c_rows[pos]][col] += coeff
            col += 1
        return col
    
    # PART 1: 4x4 x 4x4 via Strassen (rank 49)
    # A blocks: A11=(0,0),(0,1),(1,0),(1,1); A12=(0,2),(0,3),(1,2),(1,3)
    #           A21=(2,0),(2,1),(3,0),(3,1); A22=(2,2),(2,3),(3,2),(3,3)
    # B blocks: B11=(0,0),(0,1),(1,0),(1,1); B12=(0,2),(0,3),(1,2),(1,3)
    #           B21=(2,0),(2,1),(3,0),(3,1); B22=(2,2),(2,3),(3,2),(3,3)
    # C blocks: C11=(0,0),(0,1),(1,0),(1,1); C12=(0,2),(0,3),(1,2),(1,3)
    #           C21=(2,0),(2,1),(3,0),(3,1); C22=(2,2),(2,3),(3,2),(3,3)
    
    # Strassen formulas for 4x4*4x4:
    # S1 = (A11+A22)(B11+B22) -> C11+C22
    # S2 = (A21+A22)(B11) -> C21-C22
    # S3 = A11(B21-B22) -> C12+C22
    # S4 = A22(B12-B11) -> C11-C22
    # S5 = (A11+A12)(B22) -> -C11+C12
    # S6 = (A21-A11)(B11+B12) -> C22
    # S7 = (A12-A22)(B21+B22) -> C11
    
    # For each Si, we need to define the 2x2 blocks and apply Strassen.
    # The challenge: each "element" of the block is a sum of A entries.
    # We handle this by creating virtual rows.
    
    # Actually, the correct way: each Si is a product of two 2x2 matrices.
    # The left 2x2 matrix has entries that are linear combinations of A entries.
    # The right 2x2 matrix has entries that are linear combinations of B entries.
    # We apply the 2x2*2x2 Strassen to get 7 rank-1 terms.
    # For each rank-1 term, the U vector is the product of the left factor's row
    # linear combination with the right factor's column linear combination.
    
    # Let me define a helper that takes explicit linear combinations.
    
    def apply_strassen_general(a_entries, b_entries, c_entries, col):
        """
        Apply Strassen 2x2*2x2 where each entry is a linear combination.
        a_entries: list of 4 lists, each is [(u_row, coeff), ...] for A[0,0],A[0,1],A[1,0],A[1,1]
        b_entries: list of 4 lists, each is [(v_col, coeff), ...] for B[0,0],B[0,1],B[1,0],B[1,1]
        c_entries: list of 4 lists, each is [(w_row, coeff), ...] for C[0,0],C[0,1],C[1,0],C[1,1]
        """
        for u_mask, v_mask, w_mask in strassen:
            # Build U contributions
            u_contrib = {}
            for pos, coeff in u_mask.items():
                for row, c in a_entries[pos]:
                    u_contrib[row] = u_contrib.get(row, 0) + coeff * c
            # Build V contributions
            v_contrib = {}
            for pos, coeff in v_mask.items():
                for col_idx, c in b_entries[pos]:
                    v_contrib[col_idx] = v_contrib.get(col_idx, 0) + coeff * c
            # Build W contributions
            w_contrib = {}
            for pos, coeff in w_mask.items():
                for row, c in c_entries[pos]:
                    w_contrib[row] = w_contrib.get(row, 0) + coeff * c
            
            for row, val in u_contrib.items():
                if val != 0:
                    U[row][col] += val
            for col_idx, val in v_contrib.items():
                if val != 0:
                    V[col_idx][col] += val
            for row, val in w_contrib.items():
                if val != 0:
                    W[row][col] += val
            col += 1
        return col
    
    # Define block entries as lists of (index, coeff)
    # A11: A[0,0],A[0,1],A[1,0],A[1,1]
    A11 = [[(u_idx(0,0),1)], [(u_idx(0,1),1)], [(u_idx(1,0),1)], [(u_idx(1,1),1)]]
    A12 = [[(u_idx(0,2),1)], [(u_idx(0,3),1)], [(u_idx(1,2),1)], [(u_idx(1,3),1)]]
    A21 = [[(u_idx(2,0),1)], [(u_idx(2,1),1)], [(u_idx(3,0),1)], [(u_idx(3,1),1)]]
    A22 = [[(u_idx(2,2),1)], [(u_idx(2,3),1)], [(u_idx(3,2),1)], [(u_idx(3,3),1)]]
    
    B11 = [[(v_idx(0,0),1)], [(v_idx(0,1),1)], [(v_idx(1,0),1)], [(v_idx(1,1),1)]]
    B12 = [[(v_idx(0,2),1)], [(v_idx(0,3),1)], [(v_idx(1,2),1)], [(v_idx(1,3),1)]]
    B21 = [[(v_idx(2,0),1)], [(v_idx(2,1),1)], [(v_idx(3,0),1)], [(v_idx(3,1),1)]]
    B22 = [[(v_idx(2,2),1)], [(v_idx(2,3),1)], [(v_idx(3,2),1)], [(v_idx(3,3),1)]]
    
    C11 = [[(w_idx(0,0),1)], [(w_idx(0,1),1)], [(w_idx(1,0),1)], [(w_idx(1,1),1)]]
    C12 = [[(w_idx(0,2),1)], [(w_idx(0,3),1)], [(w_idx(1,2),1)], [(w_idx(1,3),1)]]
    C21 = [[(w_idx(2,0),1)], [(w_idx(2,1),1)], [(w_idx(3,0),1)], [(w_idx(3,1),1)]]
    C22 = [[(w_idx(2,2),1)], [(w_idx(2,3),1)], [(w_idx(3,2),1)], [(w_idx(3,3),1)]]
    
    def add_blocks(X, Y, sign_y=1):
        """Add two block entry lists element-wise. Y gets multiplied by sign_y."""
        result = []
        for i in range(4):
            combined = list(X[i])
            for row, c in Y[i]:
                # Check if row already exists in combined
                found = False
                for j in range(len(combined)):
                    if combined[j][0] == row:
                        combined[j] = (row, combined[j][1] + sign_y * c)
                        found = True
                        break
                if not found:
                    combined.append((row, sign_y * c))
            result.append(combined)
        return result
    
    def scale_block(X, s):
        """Scale all coefficients in a block entry list."""
        return [[(row, c * s) for row, c in X[i]] for i in range(4)]
    
    # S1 = (A11+A22)(B11+B22) -> C11+C22
    s1_a = add_blocks(A11, A22)
    s1_b = add_blocks(B11, B22)
    s1_c = add_blocks(C11, C22)
    col = apply_strassen_general(s1_a, s1_b, s1_c, col)
    
    # S2 = (A21+A22)(B11) -> C21-C22
    s2_a = add_blocks(A21, A22)
    s2_b = B11
    s2_c = add_blocks(C21, C22, sign_y=-1)
    col = apply_strassen_general(s2_a, s2_b, s2_c, col)
    
    # S3 = A11(B21-B22) -> C12+C22
    s3_a = A11
    s3_b = add_blocks(B21, B22, sign_y=-1)
    s3_c = add_blocks(C12, C22)
    col = apply_strassen_general(s3_a, s3_b, s3_c, col)
    
    # S4 = A22(B12-B11) -> C11-C22
    s4_a = A22
    s4_b = add_blocks(B12, B11, sign_y=-1)
    s4_c = add_blocks(C11, C22, sign_y=-1)
    col = apply_strassen_general(s4_a, s4_b, s4_c, col)
    
    # S5 = (A11+A12)(B22) -> -C11+C12
    s5_a = add_blocks(A11, A12)
    s5_b = B22
    s5_c = add_blocks(C11, C12, sign_y=-1)
    col = apply_strassen_general(s5_a, s5_b, s5_c, col)
    
    # S6 = (A21-A11)(B11+B12) -> C22
    s6_a = add_blocks(A21, A11, sign_y=-1)
    s6_b = add_blocks(B11, B12)
    s6_c = C22
    col = apply_strassen_general(s6_a, s6_b, s6_c, col)
    
    # S7 = (A12-A22)(B21+B22) -> C11
    s7_a = add_blocks(A12, A22, sign_y=-1)
    s7_b = add_blocks(B21, B22)
    s7_c = C11
    col = apply_strassen_general(s7_a, s7_b, s7_c, col)
    
    # PART 2: 4x4 x 4x2 (B cols 4-5, C cols 4-5): rank 28
    # A blocks same as before: A11, A12, A21, A22
    # B blocks: B_top = B[0:2, 4:6], B_bot = B[2:4, 4:6]
    # C blocks: C_top = C[0:2, 4:6], C_bot = C[2:4, 4:6]
    
    # C_top = A11*B_top + A12*B_bot
    # C_bot = A21*B_top + A22*B_bot
    
    # 4 Strassen multiplications:
    # P1 = A11 * B_top -> C_top
    # P2 = A12 * B_bot -> C_top
    # P3 = A21 * B_top -> C_bot
    # P4 = A22 * B_bot -> C_bot
    
    B_top = [[(v_idx(0,4),1)], [(v_idx(0,5),1)], [(v_idx(1,4),1)], [(v_idx(1,5),1)]]
    B_bot = [[(v_idx(2,4),1)], [(v_idx(2,5),1)], [(v_idx(3,4),1)], [(v_idx(3,5),1)]]
    
    C_top = [[(w_idx(4,0),1)], [(w_idx(5,0),1)], [(w_idx(4,1),1)], [(w_idx(5,1),1)]]
    C_bot = [[(w_idx(4,2),1)], [(w_idx(5,2),1)], [(w_idx(4,3),1)], [(w_idx(5,3),1)]]
    
    # P1 = A11 * B_top -> C_top
    col = apply_strassen_general(A11, B_top, C_top, col)
    
    # P2 = A12 * B_bot -> C_top
    col = apply_strassen_general(A12, B_bot, C_top, col)
    
    # P3 = A21 * B_top -> C_bot
    col = apply_strassen_general(A21, B_top, C_bot, col)
    
    # P4 = A22 * B_bot -> C_bot
    col = apply_strassen_general(A22, B_bot, C_bot, col)
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    elapsed = time.time() - start_time
    return {"U": U, "V": V, "W": W}
