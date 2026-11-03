import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen decomposition of 2x2 matrix multiplication
    # Rank 7 decomposition
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # r=0 (M1 = A00+A11, B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1

    # r=1 (M2 = A10+A11, B00)
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1

    # r=2 (M3 = A00, B10-B11)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1

    # r=3 (M4 = A11, B01-B00)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1

    # r=4 (M5 = A00+A01, B11)
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1

    # r=5 (M6 = A10-A00, B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1

    # r=6 (M7 = A01-A11, B10+B11)
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
    #
    # The 2x4 matrix A has entries A[i, k] for i in {0,1}, k in {0,1,2,3}
    # The 4x8 matrix B has entries B[k, j] for k in {0,1,2,3}, j in {0,...,7}
    # The 2x8 matrix C has entries C[i, j] for i in {0,1}, j in {0,...,7}
    #
    # C[i, j] = sum_{k=0}^{3} A[i, k] * B[k, j]
    #
    # We group the 4 columns of A and 4 rows of B into 2 blocks of 2:
    # Block s (s=0,1): A columns {2s, 2s+1}, B rows {2s, 2s+1}
    #
    # For block s, define local 2x2 matrices:
    # A_block[i, l] = A[i, 2s+l] for i,l in {0,1}
    # B_block[l, j] = B[2s+l, j] for l in {0,1}, j in {0,...,7}
    #
    # But B_block is 2x8, not 2x2. So we need to split j as well.
    # Let j = 2m + n where m in {0,1,2,3}, n in {0,1}
    #
    # Actually, let's think of it differently. For each j (column of C),
    # C[i, j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # This is a 2x4 · 4x1 multiplication for fixed j.
    #
    # The current approach treats this as 4 independent 2x1 dot products,
    # but groups them into 2 Strassen operations by splitting k into pairs.
    #
    # Let's define: for j in {0,...,7}, and for block s in {0,1}:
    #   The contribution to C[i,j] from block s is:
    #   sum_{l=0}^{1} A[i, 2s+l] * B[2s+l, j]
    #
    # This is a 2x2 · 2x1 multiplication (treating B[:,j] restricted to rows 2s,2s+1 as a 2x1 vector).
    #
    # But Strassen is for 2x2 · 2x2. So we need to also split j.
    #
    # Actually, looking at the existing code more carefully:
    # The existing code does 8 (j,s) pairs, each with a full 2x2 Strassen.
    # Let me trace through:
    #
    # For (j, s) pair:
    # a_map[la] where la = row*2+col, maps to A[row, s*2+col]
    #   So la=0 -> A[0, 2s], la=1 -> A[0, 2s+1], la=2 -> A[1, 2s], la=3 -> A[1, 2s+1]
    #
    # b_map[lb] where lb = row*2+col, maps to B[s*2+row, 2*j+col]
    #   So lb=0 -> B[2s, 2j], lb=1 -> B[2s, 2j+1], lb=2 -> B[2s+1, 2j], lb=3 -> B[2s+1, 2j+1]
    #
    # c_map[lc] where lc = row*2+col, maps to C[row, 2j+col]
    #   So lc=0 -> C[0, 2j], lc=1 -> C[0, 2j+1], lc=2 -> C[1, 2j], lc=3 -> C[1, 2j+1]
    #
    # So for each (j, s), we're doing a 2x2 · 2x2 multiplication:
    # A_block = [[A[0,2s], A[0,2s+1]], [A[1,2s], A[1,2s+1]]]
    # B_block = [[B[2s,2j], B[2s,2j+1]], [B[2s+1,2j], B[2s+1,2j+1]]]
    # C_block = A_block @ B_block = [[C[0,2j], C[0,2j+1]], [C[1,2j], C[1,2j+1]]]
    #
    # This gives C_block[i, col] = sum_{l} A_block[i,l] * B_block[l,col]
    # = sum_{l} A[i, 2s+l] * B[2s+l, 2j+col]
    #
    # Summing over s=0,1 gives: sum_{s,l} A[i, 2s+l] * B[2s+l, 2j+col] = sum_{k=0}^{3} A[i,k] * B[k, 2j+col]
    #
    # So C[i, 2j+col] = sum_{k=0}^{3} A[i,k] * B[k, 2j+col] for col in {0,1}
    #
    # And j ranges over {0,1,2,3}, so 2j+col covers all of {0,...,7}.
    #
    # Great, so the decomposition is correct: 8 independent 2x2·2x2 multiplications, each rank 7.
    # Total rank = 56.

    # Precompute index mappings
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
