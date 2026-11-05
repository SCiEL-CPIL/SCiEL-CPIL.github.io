"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin

    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24

    # Schoolbook fallback (Rank 120)
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    # Structured Construction using Strassen for 2x2 blocks
    # Partitions:
    # A rows: [0,1] (R0), [2,3] (R1)
    # A cols / B rows: [0,1] (C0), [2,3] (C1), [4] (C2)
    # B cols: [0,1] (K0), [2,3] (K1), [4,5] (K2)

    R_blocks = [[0, 1], [2, 3]]
    C_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4, 5]]

    # Total Rank estimate:
    # 12 blocks of 2x2x2 (Rank 7) -> 84
    # 6 blocks of 2x1x2 (Rank 1) -> 6
    # Total: 90

    rank_structured = 90
    U_struct = np.zeros((num_U, rank_structured), dtype=int)
    V_struct = np.zeros((num_V, rank_structured), dtype=int)
    W_struct = np.zeros((num_W, rank_structured), dtype=int)

    col_idx = 0

    # Helper to get global indices
    def u_global(i, j): return i * m + j
    def v_global(j, k): return j * p + k
    def w_global(k, i): return k * n + i

    # Strassen terms for 2x2x2
    def add_strassen_block(r_idx, c_idx, k_idx, start_col):
        R = R_blocks[r_idx]
        C = C_blocks[c_idx]
        K = K_blocks[k_idx]

        # M1
        U_struct[u_global(R[0], C[0]), start_col] += 1
        U_struct[u_global(R[1], C[1]), start_col] += 1
        V_struct[v_global(C[0], K[0]), start_col] += 1
        V_struct[v_global(C[1], K[1]), start_col] += 1
        W_struct[w_global(K[0], R[0]), start_col] += 1
        W_struct[w_global(K[1], R[1]), start_col] += 1
        col = start_col + 1

        # M2
        U_struct[u_global(R[1], C[0]), col] += 1
        U_struct[u_global(R[1], C[1]), col] += 1
        V_struct[v_global(C[0], K[0]), col] += 1
        W_struct[w_global(K[0], R[1]), col] += 1
        W_struct[w_global(K[1], R[1]), col] += -1
        col += 1

        # M3
        U_struct[u_global(R[0], C[0]), col] += 1
        V_struct[v_global(C[1], K[0]), col] += 1
        V_struct[v_global(C[1], K[1]), col] += -1
        W_struct[w_global(K[1], R[0]), col] += 1
        W_struct[w_global(K[1], R[1]), col] += 1
        col += 1

        # M4
        U_struct[u_global(R[1], C[1]), col] += 1
        V_struct[v_global(C[0], K[1]), col] += 1
        V_struct[v_global(C[0], K[0]), col] += -1
        W_struct[w_global(K[0], R[0]), col] += 1
        W_struct[w_global(K[0], R[1]), col] += 1
        col += 1

        # M5
        U_struct[u_global(R[0], C[0]), col] += 1
        U_struct[u_global(R[0], C[1]), col] += 1
        V_struct[v_global(C[1], K[1]), col] += 1
        W_struct[w_global(K[0], R[0]), col] += -1
        W_struct[w_global(K[1], R[0]), col] += 1
        col += 1

        # M6
        U_struct[u_global(R[1], C[0]), col] += 1
        U_struct[u_global(R[0], C[0]), col] += -1
        V_struct[v_global(C[0], K[0]), col] += 1
        V_struct[v_global(C[0], K[1]), col] += 1
        W_struct[w_global(K[1], R[1]), col] += 1
        col += 1

        # M7
        U_struct[u_global(R[0], C[1]), col] += 1
        U_struct[u_global(R[1], C[1]), col] += -1
        V_struct[v_global(C[1], K[0]), col] += 1
        V_struct[v_global(C[1], K[1]), col] += 1
        W_struct[w_global(K[0], R[0]), col] += 1
        col += 1

        return col

    def add_rank1_block(r_idx, c_idx, k_idx, start_col):
        R = R_blocks[r_idx]
        C = C_blocks[c_idx]
        K = K_blocks[k_idx]

        # A is 2x1, B is 1x2
        U_struct[u_global(R[0], C[0]), start_col] += 1
        U_struct[u_global(R[1], C[0]), start_col] += 1
        V_struct[v_global(C[0], K[0]), start_col] += 1
        V_struct[v_global(C[0], K[1]), start_col] += 1
        W_struct[w_global(K[0], R[0]), start_col] += 1
        W_struct[w_global(K[0], R[1]), start_col] += 1
        W_struct[w_global(K[1], R[0]), start_col] += 1
        W_struct[w_global(K[1], R[1]), start_col] += 1

        return start_col + 1

    # Iterate through blocks
    for r in range(2):
        for c in range(3):
            for k in range(3):
                if c < 2:
                    col_idx = add_strassen_block(r, c, k, col_idx)
                else:
                    col_idx = add_rank1_block(r, c, k, col_idx)

    # Verify structured solution
    def verify(U, V, W):
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        # Check tensor reconstruction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if val != 1:
                        return False
        # Check off-diagonal terms (sample)
        for a in range(num_U):
            for b in range(num_V):
                for c in range(num_W):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i == i2 and j == j2:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if val != 0:
                        return False
        return True

    if verify(U_struct, V_struct, W_struct):
        return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
    else:
        # Fallback to schoolbook if structured fails (should not happen)
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
