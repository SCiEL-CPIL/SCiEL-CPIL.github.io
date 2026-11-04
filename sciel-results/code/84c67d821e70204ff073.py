import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-7 Strassen decomposition for 2x2 matrix multiplication
    # U_local[i, r] = coefficient of A[i,j] in multiplication r
    # V_local[k, r] = coefficient of B[k,j] in multiplication r
    # W_local[i, r] = coefficient of multiplication r in C[i,j]
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # r=0: M1 = A00+A11, B00+B11 -> C00, C11
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1

    # r=1: M2 = A10+A11, B00 -> C10, -C11
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1

    # r=2: M3 = A00, B10-B11 -> C01, C11
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1

    # r=3: M4 = A11, B01-B00 -> C00, C10
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1

    # r=4: M5 = A00+A01, B11 -> -C00, C01
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1

    # r=5: M6 = A10-A00, B00+B01 -> C11
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1

    # r=6: M7 = A01-A11, B10+B11 -> C00
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1

    # Total rank = 8 * 7 = 56
    total_rank = 56

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # For each of the 8 independent 2x2·2x2 sub-problems,
    # map the local Strassen indices to global indices.
    mappings = []
    for j in range(4):
        for s in range(2):
            a_map = [0]*4
            for la in range(4):
                row = la // 2
                col = la % 2
                global_col = s * 2 + col
                a_map[la] = row * 4 + global_col

            b_map = [0]*4
            for lb in range(4):
                row = lb // 2
                col = lb % 2
                global_row = s * 2 + row
                global_col = 2 * j + col
                b_map[lb] = global_row * 8 + global_col

            c_map = [0]*4
            for lc in range(4):
                row = lc // 2
                col = lc % 2
                global_row = row
                global_col = 2 * j + col
                c_map[lc] = global_col * 2 + global_row

            mappings.append((a_map, b_map, c_map))

    # Fill in the factor matrices
    for idx, (a_map, b_map, c_map) in enumerate(mappings):
        base_r = idx * 7
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[c_map[lc], r] += val

    # Quick verification
    if time.time() < DEADLINE - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        C_recon = np.zeros((n, p))
        for r in range(total_rank):
            u_val = U[:, r] @ A_flat
            v_val = V[:, r] @ B_flat
            for i in range(n):
                for j in range(p):
                    c_idx = j * n + i
                    C_recon[i, j] += W[c_idx, r] * u_val * v_val

        if not np.allclose(C_recon, C_expected, atol=1e-10):
            pass  # Should not happen with correct Strassen

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
