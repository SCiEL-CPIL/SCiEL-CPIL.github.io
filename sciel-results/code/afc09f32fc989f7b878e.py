import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n, m, p = 4, 4, 9

    def time_ok():
        return time.time() < deadline

    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    U_222 = np.zeros((4, 7), dtype=float)
    V_222 = np.zeros((4, 7), dtype=float)
    W_222 = np.zeros((4, 7), dtype=float)

    # s1 = (a11+a22)(b11+b22)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1

    # s2 = (a21+a22)b11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1

    # s3 = a11(b21-b11)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1

    # s4 = a22(b12-b22)
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[3, 3] = 1

    # s5 = (a11+a12)b22
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = -1

    # s6 = (a21-a11)(b11+b12)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[2, 5] = 1

    # s7 = (a12-a22)(b21+b22)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[1, 6] = 1

    # 2. Construct 4x4x4 Strassen decomposition (Rank 49)
    U_444 = np.zeros((16, 49), dtype=float)
    V_444 = np.zeros((16, 49), dtype=float)
    W_444 = np.zeros((16, 49), dtype=float)

    def add_block_term(U_target, V_target, W_target,
                       a_blocks, b_blocks, c_blocks, col_offset):
        for s in range(7):
            col = col_offset + s
            for (br, bc, sign) in a_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        u_idx = i_global * 4 + j_global
                        u_local_idx = i_local * 2 + j_local
                        U_target[u_idx, col] += sign * U_222[u_local_idx, s]
            for (br, bc, sign) in b_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        v_idx = i_global * 4 + j_global
                        v_local_idx = i_local * 2 + j_local
                        V_target[v_idx, col] += sign * V_222[v_local_idx, s]
            for (br, bc, sign) in c_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        w_idx = i_global * 4 + j_global
                        w_local_idx = i_local * 2 + j_local
                        W_target[w_idx, col] += sign * W_222[w_local_idx, s]

    add_block_term(U_444, V_444, W_444,
                   [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0)
    add_block_term(U_444, V_444, W_444,
                   [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)], 7)
    add_block_term(U_444, V_444, W_444,
                   [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 14)
    add_block_term(U_444, V_444, W_444,
                   [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,1,1)], 21)
    add_block_term(U_444, V_444, W_444,
                   [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,-1)], 28)
    add_block_term(U_444, V_444, W_444,
                   [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1)], 35)
    add_block_term(U_444, V_444, W_444,
                   [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,1)], 42)

    # 3. Construct 4x4x1 Schoolbook decomposition (Rank 16)
    U_441 = np.zeros((16, 16), dtype=float)
    V_441 = np.zeros((4, 16), dtype=float)
    W_441 = np.zeros((4, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1

    # 4. Assemble full 4x4x9 decomposition
    total_rank = 114
    U_total = np.zeros((16, total_rank), dtype=float)
    V_total = np.zeros((36, total_rank), dtype=float)
    W_total = np.zeros((36, total_rank), dtype=float)

    # Part 1: C1 = A B1 (cols 0-3 of B, cols 0-3 of C)
    U_total[:, 0:49] = U_444
    V_part1 = np.zeros((36, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part1[j*9 + k, :] = V_444[j*4 + k, :]
    V_total[:, 0:49] = V_part1
    W_part1 = np.zeros((36, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part1[k*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 0:49] = W_part1

    # Part 2: C2 = A B2 (cols 4-7 of B, cols 4-7 of C)
    U_total[:, 49:98] = U_444
    V_part2 = np.zeros((36, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part2[j*9 + (4+k), :] = V_444[j*4 + k, :]
    V_total[:, 49:98] = V_part2
    W_part2 = np.zeros((36, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part2[(4+k)*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 49:98] = W_part2

    # Part 3: C3 = A B3 (col 8 of B, col 8 of C)
    U_total[:, 98:114] = U_441
    V_part3 = np.zeros((36, 16), dtype=float)
    for j in range(4):
        V_part3[j*9 + 8, :] = V_441[j, :]
    V_total[:, 98:114] = V_part3
    W_part3 = np.zeros((36, 16), dtype=float)
    for i in range(4):
        W_part3[8*4 + i, :] = W_441[i, :]
    W_total[:, 98:114] = W_part3

    # 5. Fast validation
    def validate_fast(U, V, W, tol=1e-9):
        U_arr = np.asarray(U, dtype=float)
        V_arr = np.asarray(V, dtype=float)
        W_arr = np.asarray(W, dtype=float)
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        prod = U_arr[u_rows, :] * V_arr[v_rows, :] * W_arr[w_rows, :]
        sums = prod.sum(axis=1)
        return np.all(np.abs(sums - 1.0) <= tol)

    # Verify initial decomposition
    if not validate_fast(U_total, V_total, W_total):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=float)
        V_sb = np.zeros((m*p, rank_sb), dtype=float)
        W_sb = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
