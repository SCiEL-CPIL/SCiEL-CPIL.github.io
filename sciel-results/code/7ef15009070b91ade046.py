import time

DIMENSIONS = [2, 4, 4]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 21
    V = [[0] * rank for _ in range(m * p)]  # 16 x 21
    W = [[0] * rank for _ in range(p * n)]  # 8 x 21
    
    # Strassen terms for 2x2x2
    # M1: (a00+a11)(b00+b11) -> c00, c11
    # M2: (a10+a11)b00 -> c10, -c11
    # M3: a00(b01-b11) -> c01, c11
    # M4: a11(b10-b00) -> c00, c10
    # M5: (a00+a01)b11 -> -c00, c01
    # M6: (a10-a00)(b00+b01) -> c11
    # M7: (a01-a11)(b10+b11) -> c00
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Three blocks:
    # Block 1: A_row0 cols 0-1, B rows 0-1 cols 0-1, C_row0 cols 0-1
    # Block 2: A_row0 cols 2-3, B rows 2-3 cols 0-1, C_row0 cols 2-3
    # Block 3: A_row1 cols 0-3, B rows 0-1 cols 2-3, C_row1 cols 2-3
    # Wait, let me reconsider. B is 4x4.
    # C = A * B, A is 2x4, B is 4x4, C is 2x4
    # C[0][j] = sum_k A[0][k] * B[k][j]
    # C[1][j] = sum_k A[1][k] * B[k][j]
    #
    # For row 0 of C:
    #   C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0]
    #   C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1]
    #   C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2]
    #   C[0][3] = A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3]
    #
    # Split into two 2x2 blocks for row 0:
    #   Block 1: A[0][0:2] x B[0:2, 0:2] -> C[0][0:2]
    #   Block 2: A[0][2:4] x B[2:4, 0:2] -> C[0][0:2]
    #   Block 3: A[0][0:2] x B[0:2, 2:4] -> C[0][2:4]
    #   Block 4: A[0][2:4] x B[2:4, 2:4] -> C[0][2:4]
    #
    # That's 4 blocks for row 0 alone. For row 1 similarly 4 blocks. Total 8 blocks x 7 = 56?
    # No wait, that's worse than 28.
    #
    # Let me reconsider the original approach. The original used:
    # Block 1: A_L (cols 0,1), B_TL (rows 0,1 cols 0,1), C_L (cols 0,1)
    # Block 2: A_R (cols 2,3), B_BL (rows 2,3 cols 0,1), C_L (cols 0,1)
    # Block 3: A_L (cols 0,1), B_TR (rows 0,1 cols 2,3), C_R (cols 2,3)
    # Block 4: A_R (cols 2,3), B_BR (rows 2,3 cols 2,3), C_R (cols 2,3)
    #
    # Each block is a 2x2 x 2x2 -> 2x2 multiplication using 7 Strassen terms.
    # 4 blocks x 7 = 28.
    #
    # The key realization: A has only 2 rows. In each block, A is 2x2 (both rows, 2 cols).
    # So we can't reduce below 4 blocks if we want to cover all of A.
    #
    # But wait - maybe we can use a different decomposition. What if we use rank-5 algorithms
    # for 2x2x2? No, 7 is the known minimum for 2x2x2.
    #
    # Let me think differently. The tensor is 2x4x4. What's the known rank?
    # It's at most 28 (this solution). Can we do better?
    #
    # Actually, let me just verify the current solution works and return it.
    # The improvement would come from finding a better decomposition.
    
    blocks = [
        # (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
        (0, 0, 0, 0, 0, 0), # Block 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0), # Block 2: A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 0, 2), # Block 3: A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 0, 2)  # Block 4: A[0:2,2:4] x B[2:4,2:4] -> C[0:2,2:4]
    ]
    
    col_idx = 0
    for a_row_s, a_col_s, b_row_s, b_col_s, c_row_s, c_col_s in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U: A has 2 rows, 4 cols. Local 0..3 = A[0,a_col_s], A[0,a_col_s+1], A[1,a_col_s], A[1,a_col_s+1]
            for local_idx, val in u_coeffs.items():
                if local_idx == 0:
                    r, c = 0, a_col_s
                elif local_idx == 1:
                    r, c = 0, a_col_s + 1
                elif local_idx == 2:
                    r, c = 1, a_col_s
                elif local_idx == 3:
                    r, c = 1, a_col_s + 1
                else:
                    continue
                g_idx = r * m + c
                U[g_idx][col_idx] = val
            
            # Fill V: B has 4 rows, 4 cols. Local 0..3 = B[b_row_s,b_col_s], B[b_row_s,b_col_s+1], B[b_row_s+1,b_col_s], B[b_row_s+1,b_col_s+1]
            for local_idx, val in v_coeffs.items():
                if local_idx == 0:
                    r, c = b_row_s, b_col_s
                elif local_idx == 1:
                    r, c = b_row_s, b_col_s + 1
                elif local_idx == 2:
                    r, c = b_row_s + 1, b_col_s
                elif local_idx == 3:
                    r, c = b_row_s + 1, b_col_s + 1
                else:
                    continue
                g_idx = r * p + c
                V[g_idx][col_idx] = val
            
            # Fill W: C has 2 rows, 4 cols. W[k*n+i, r] where k=col, i=row
            # Local 0..3 = C[c_row_s, c_col_s], C[c_row_s, c_col_s+1], C[c_row_s+1, c_col_s], C[c_row_s+1, c_col_s+1]
            for local_idx, val in w_coeffs.items():
                if local_idx == 0:
                    r, c = c_row_s, c_col_s
                elif local_idx == 1:
                    r, c = c_row_s, c_col_s + 1
                elif local_idx == 2:
                    r, c = c_row_s + 1, c_col_s
                elif local_idx == 3:
                    r, c = c_row_s + 1, c_col_s + 1
                else:
                    continue
                # W[k*n+i] where k=col index, i=row index
                g_idx = c * n + r
                W[g_idx][col_idx] = val
            
            col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
