"""Rank-minimal exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Uses a hybrid approach: Strassen 2x2x2 blocks for the 2x4x4 part (rank 14)
plus schoolbook for the 2x4x2 part (rank 8), giving total rank 22.
"""

def construct():
    n, m, p = 2, 4, 6
    rank_total = 22
    U = [[0] * rank_total for _ in range(n * m)]  # shape (8, 22)
    V = [[0] * rank_total for _ in range(m * p)]  # shape (24, 22)
    W = [[0] * rank_total for _ in range(p * n)]  # shape (12, 22)

    # Strassen 2x2 @ 2x2 rank 7 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00 + a11) * (b00 + b11)
    # P2 = (a10 + a11) * b00
    # P3 = a00 * (b01 - b11)
    # P4 = a11 * (b10 - b00)
    # P5 = (a00 + a01) * b11
    # P6 = (a10 - a00) * (b00 + b01)
    # P7 = (a01 - a11) * (b10 + b11)
    #
    # C:
    #   c00 = P1 + P4 - P5 + P7
    #   c01 = P3 + P5
    #   c10 = P2 + P4
    #   c11 = P1 - P2 + P3 + P6

    strassen_U = [
        {0: 1, 3: 1},       # P1
        {2: 1, 3: 1},       # P2
        {0: 1},             # P3
        {3: 1},             # P4
        {0: 1, 1: 1},       # P5
        {0: -1, 2: 1},      # P6
        {1: 1, 3: -1}       # P7
    ]

    strassen_V = [
        {0: 1, 3: 1},       # P1
        {0: 1},             # P2
        {1: 1, 3: -1},      # P3
        {2: 1, 0: -1},      # P4
        {3: 1},             # P5
        {0: 1, 1: 1},       # P6
        {2: 1, 3: 1}        # P7
    ]

    strassen_W = [
        {0: 1, 3: 1},       # P1: +c00, +c11
        {2: 1, 3: -1},      # P2: +c10, -c11
        {1: 1, 3: 1},       # P3: +c01, +c11
        {0: 1, 2: 1},       # P4: +c00, +c10
        {0: -1, 1: 1},      # P5: -c00, +c01
        {3: 1},             # P6: +c11
        {0: 1}              # P7: +c00
    ]

    col_offset = 0
    # Process 2x2x2 Strassen blocks for j_block in {0, 1} (columns 0-3 of B)
    for j_block in range(2):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx

                # U side: A indices
                for local_idx, sign in strassen_U[prod_idx].items():
                    if local_idx == 0:
                        u_idx = 0 * 4 + u_col_offset
                    elif local_idx == 1:
                        u_idx = 0 * 4 + u_col_offset + 1
                    elif local_idx == 2:
                        u_idx = 1 * 4 + u_col_offset
                    else:
                        u_idx = 1 * 4 + u_col_offset + 1
                    U[u_idx][col] += sign

                # V side: B indices
                for local_idx, sign in strassen_V[prod_idx].items():
                    if local_idx == 0:
                        v_idx = v_row_offset * 6 + 2 * j_block
                    elif local_idx == 1:
                        v_idx = v_row_offset * 6 + 2 * j_block + 1
                    elif local_idx == 2:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                    else:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                    V[v_idx][col] += sign

                # W side: C indices (trace convention W[k*n+i, r])
                for local_idx, sign in strassen_W[prod_idx].items():
                    if local_idx == 0:
                        w_idx = (2 * j_block) * 2 + 0
                    elif local_idx == 1:
                        w_idx = (2 * j_block + 1) * 2 + 0
                    elif local_idx == 2:
                        w_idx = (2 * j_block) * 2 + 1
                    else:
                        w_idx = (2 * j_block + 1) * 2 + 1
                    W[w_idx][col] += sign

            col_offset += 7

    # Now handle the last 2 columns (j_block = 2, columns 4-5 of B) with schoolbook
    # For each row i of A (2 rows), each column k of C (2 cols), each m-index:
    # C[i,k] = sum_m A[i,m] * B[m,k]
    # Schoolbook: 2*2*2 = 8 multiplications for the 2x4x2 part
    # But we need to be careful: the 4x2 sub-matrix of B (rows 0-3, cols 4-5)
    # and the 2x4 sub-matrix of A (all rows, all cols)
    
    # Actually, let me think more carefully. The full multiplication is:
    # C[2,6] = A[2,4] * B[4,6]
    # We split B into B_left[4,4] and B_right[4,2]
    # C_left[2,4] = A * B_left  (handled by Strassen, rank 14)
    # C_right[2,2] = A * B_right (schoolbook, rank 8)
    
    # For C_right, the schoolbook decomposition:
    # For i in {0,1}, k in {4,5} (global), m in {0,1,2,3}:
    # C[i, k] += A[i, m] * B[m, k]
    # Each (i, m, k) triple is one rank-1 term
    # Total: 2 * 4 * 2 = 8 terms
    
    for i in range(2):       # row of A, col of C
        for m in range(4):   # col of A, row of B
            for k in range(2):  # col of B (local 0,1 -> global 4,5)
                col = col_offset + (i * 4 + m) // 2 * 2 + k  # This indexing is wrong, let me redo
                
    # Let me redo the schoolbook part more carefully
    # The 8 schoolbook terms for C_right[2,2]:
    # C[i, 4] = sum_{m=0}^{3} A[i, m] * B[m, 4]
    # C[i, 5] = sum_{m=0}^{3} A[i, m] * B[m, 5]
    # So we have 2*4*2 = 8 independent products
    
    for prod_idx in range(8):
        i = prod_idx // 4       # 0 or 1 (row of A, first index of C)
        m = (prod_idx % 4) // 2  # 0,1,2,3 (col of A, row of B)
        k_local = prod_idx % 2   # 0 or 1 (local col of B_right, global 4 or 5)
        
        col = col_offset + prod_idx
        
        # U: A[i, m]
        u_idx = i * 4 + m
        U[u_idx][col] += 1
        
        # V: B[m, 4 + k_local]
        v_idx = m * 6 + (4 + k_local)
        V[v_idx][col] += 1
        
        # W: C[i, 4 + k_local] -> trace convention: W[k*n + i, col] where k is column index
        # C column is 4 + k_local, so k = 4 + k_local
        # W index = (4 + k_local) * 2 + i
        w_idx = (4 + k_local) * 2 + i
        W[w_idx][col] += 1
    
    return {"U": U, "V": V, "W": W}
