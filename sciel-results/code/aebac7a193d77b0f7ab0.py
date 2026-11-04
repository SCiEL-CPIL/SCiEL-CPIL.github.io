import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Strassen 2x2x2 decomposition (7 terms)
    # U_2: 4x7, V_2: 4x7, W_2: 4x7
    # Index: r*2+c for 2x2 matrix position
    U_2 = np.zeros((4, 7), dtype=np.int8)
    V_2 = np.zeros((4, 7), dtype=np.int8)
    W_2 = np.zeros((4, 7), dtype=np.int8)

    # c=0: (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: (a10+a11)b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: a00(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: a11(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: (a00+a01)b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: (a10-a00)(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: (a01-a11)(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen for 4x4x4:
    # A is 4x4, split into 2x2 blocks of size 2x2
    # A00: rows 0-1, cols 0-1 (block 0)
    # A01: rows 0-1, cols 2-3 (block 1)
    # A10: rows 2-3, cols 0-1 (block 2)
    # A11: rows 2-3, cols 2-3 (block 3)
    #
    # For each half (L=0, R=1), B is 4x4 (cols 0-3 or 4-7 of the 4x8 matrix)
    # B blocks similarly indexed.
    #
    # 7 products P_k:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    #
    # C = 4x4 (for each half):
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5

    # A_specs[k_p]: list of (coeff, block_idx) where block_idx in {0,1,2,3}
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]

    # B_specs[k_p]: list of (coeff, block_idx)
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]

    # C_specs[k_p]: list of (coeff, c_block_idx)
    # C00=0, C01=1, C10=2, C11=3
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]

    col = 0
    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                # --- U (Input A, shape 16x98) ---
                # A index: i*4+j where i,j in 0..3
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            val = coeff_a * int(U_2[a_eff_idx, m_s])
                            if val != 0:
                                U[global_a_idx, col] += val

                # --- V (Input B, shape 32x98) ---
                # B index: j*8+k where j in 0..3, k in 0..7
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            val = coeff_b * int(V_2[b_eff_idx, m_s])
                            if val != 0:
                                V[global_b_idx, col] += val

                # --- W (Output C, shape 32x98) ---
                # W index: k*4+i where k in 0..7, i in 0..3
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            val = coeff_c * int(W_2[p_eff_idx, m_s])
                            if val != 0:
                                W[global_w_idx, col] += val

                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
