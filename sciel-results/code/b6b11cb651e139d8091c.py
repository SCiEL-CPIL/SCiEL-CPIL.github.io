"""Matrix multiplication 2x4 by 4x4, rank-minimal decomposition via Strassen blocks."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen for 2x2 * 2x2 -> 2x2
    # P1 = (A00+A01)(B00+B10)
    # P2 = (A10+A11)(B00+B10)
    # P3 = A00*(B10-B00)
    # P4 = A11*(B11-B01)
    # P5 = (A00+A11)(B01+B11)
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A11-A01)(B01+B11)
    # C00 = P1+P4-P5-P7
    # C01 = P5+P7
    # C10 = P1-P3+P6+P7
    # C11 = P3-P1+P2+P4

    # We have A: 2x4, B: 4x4, C: 2x4
    # Split A into two 2x2 blocks: A_left = A[:, 0:2], A_right = A[:, 2:4]
    # Split B into four 2x2 blocks: B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C_left = A_left * B_left where B_left = B[:, 0:2] (2x4)
    #   Actually, C = A * B where A is 2x4, B is 4x4, C is 2x4
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   Split j into j=0,1 and j=2,3
    #   C[i,k] = sum_{j=0,1} A[i,j] B[j,k] + sum_{j=2,3} A[i,j] B[j,k]
    #   Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    #   Let B1 = B[0:2, :] (2x4), B2 = B[2:4, :] (2x4)
    #   C = A1 * B1 + A2 * B2
    #   Now B1 is 2x4, split into B1_left = B1[:, 0:2] (2x2), B1_right = B1[:, 2:4] (2x2)
    #   B2 is 2x4, split into B2_left = B2[:, 0:2] (2x2), B2_right = B2[:, 2:4] (2x2)
    #   C_left = A1 * B1_left + A2 * B2_left (2x2)
    #   C_right = A1 * B1_right + A2 * B2_right (2x2)
    #   
    #   So we have 4 Strassen instances:
    #   Instance 1: A1 * B1_left -> contributes to C_left
    #   Instance 2: A2 * B2_left -> contributes to C_left
    #   Instance 3: A1 * B1_right -> contributes to C_right
    #   Instance 4: A2 * B2_right -> contributes to C_right
    #   Total rank = 4 * 7 = 28

    # Strassen coefficients
    # For X (left matrix, 2x2): local (r, c) where r=0,1 is row, c=0,1 is col
    # For Y (right matrix, 2x2): local (r, c) where r=0,1 is row, c=0,1 is col
    # For Z (result, 2x2): local (r, c) where r=0,1 is row, c=0,1 is col

    strassen_X = [
        {(0,0): 1, (0,1): 1},  # P1
        {(1,0): 1, (1,1): 1},  # P2
        {(0,0): 1},            # P3
        {(1,1): 1},            # P4
        {(0,0): 1, (1,1): 1},  # P5
        {(1,0): 1, (0,0): -1}, # P6
        {(1,1): 1, (0,1): -1}, # P7
    ]

    strassen_Y = [
        {(0,0): 1, (1,0): 1},  # P1
        {(0,0): 1, (1,0): 1},  # P2
        {(1,0): 1, (0,0): -1}, # P3
        {(1,1): 1, (1,0): -1}, # P4
        {(1,1): 1, (0,1): -1}, # P5
        {(0,0): 1, (0,1): 1},  # P6
        {(0,1): 1, (1,1): 1},  # P7
    ]

    strassen_Z = [
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1},  # P1: C00+=1, C11+=1, C01+=-1, C10+=-1
        {(0,1): 1, (1,1): 1},                         # P2: C01+=1, C11+=1
        {(0,0): -1, (1,0): 1},                        # P3: C00+=-1, C10+=1
        {(0,0): 1, (1,1): 1},                         # P4: C00+=1, C11+=1
        {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1},   # P5: C00+=-1, C01+=1, C10+=-1, C11+=1
        {(1,0): 1, (0,1): 1},                         # P6: C10+=1, C01+=1
        {(0,0): -1, (0,1): 1, (1,0): 1, (1,1): -1},   # P7: C00+=-1, C01+=1, C10+=1, C11+=-1
    ]

    # Wait, let me re-derive the Z coefficients more carefully.
    # C00 = P1 + P4 - P5 - P7
    # C01 = P5 + P7
    # C10 = P1 - P3 + P6 + P7
    # C11 = P3 - P1 + P2 + P4
    #
    # So for each Pk, the contribution to each C position:
    # P1: C00 += 1, C10 += 1
    # P2: C11 += 1
    # P3: C10 += -1, C11 += 1
    # P4: C00 += 1, C11 += 1
    # P5: C00 += -1, C01 += 1
    # P6: C10 += 1
    # P7: C00 += -1, C01 += 1, C10 += 1

    strassen_Z_correct = [
        {(0,0): 1, (1,0): 1},          # P1
        {(1,1): 1},                     # P2
        {(1,0): -1, (1,1): 1},         # P3
        {(0,0): 1, (1,1): 1},          # P4
        {(0,0): -1, (0,1): 1},         # P5
        {(1,0): 1},                     # P6
        {(0,0): -1, (0,1): 1, (1,0): 1} # P7
    ]

    # Now let's build the decomposition
    # Instance 1: A1 (A[:,0:2]) * B1_left (B[0:2, 0:2]) -> C_left (C[:, 0:2])
    # Instance 2: A2 (A[:,2:4]) * B2_left (B[2:4, 0:2]) -> C_left (C[:, 0:2])
    # Instance 3: A1 (A[:,0:2]) * B1_right (B[0:2, 2:4]) -> C_right (C[:, 2:4])
    # Instance 4: A2 (A[:,2:4]) * B2_right (B[2:4, 2:4]) -> C_right (C[:, 2:4])

    # A[i, j] global index in U: i*m + j = 2*i + j
    # B[j, k] global index in V: j*p + k = 4*j + k
    # C[i, k] global index in W: k*n + i = 2*k + i

    col = 0
    # (A_block_offset, B_block_offset, C_block_offset)
    # A_block: 0 = left (j=0,1), 1 = right (j=2,3)
    # B_block: 0 = top-left (j=0,1; k=0,1), 1 = top-right (j=0,1; k=2,3)
    #          2 = bottom-left (j=2,3; k=0,1), 3 = bottom-right (j=2,3; k=2,3)
    # C_block: 0 = left (k=0,1), 1 = right (k=2,3)

    instances = [
        (0, 0, 0),  # A1 * B00 -> C_left
        (1, 2, 0),  # A2 * B20 -> C_left  (B20 = B[2:4, 0:2])
        (0, 1, 1),  # A1 * B01 -> C_right (B01 = B[0:2, 2:4])
        (1, 3, 1),  # A2 * B21 -> C_right (B21 = B[2:4, 2:4])
    ]

    for (a_off, b_off, c_off) in instances:
        for p_idx in range(7):
            # X coefficients (A part)
            for (r, c), coeff in strassen_X[p_idx].items():
                # Global A index: i = r, j = a_off*2 + c
                i_global = r
                j_global = a_off * 2 + c
                U[i_global * m + j_global][col] += coeff

            # Y coefficients (B part)
            for (r, c), coeff in strassen_Y[p_idx].items():
                # Global B index: j = b_off_row*2 + r, k = b_off_col*2 + c
                # b_off encoding: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) where (row_offset, col_offset)
                b_row_off = b_off // 2
                b_col_off = b_off % 2
                j_global = b_row_off * 2 + r
                k_global = b_col_off * 2 + c
                V[j_global * p + k_global][col] += coeff

            # Z coefficients (C part)
            for (r, c), coeff in strassen_Z_correct[p_idx].items():
                # Global C index: i = r, k = c_off*2 + c
                i_global = r
                k_global = c_off * 2 + c
                W[k_global * n + i_global][col] += coeff

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
