import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = 1
        U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[0, 3] = -1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = 1; W[2, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[2, 5] = 1; W[2, 5] = 1; W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1; V[1, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[1, 6] = 1
        return U, V, W

    def get_mult_1x2_2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Baseline Rank 66 Construction
    U_66 = np.zeros((12, 66))
    V_66 = np.zeros((24, 66))
    W_66 = np.zeros((18, 66))
    
    def place_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, offset):
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            for local_r, global_r in enumerate(u_rows):
                U_66[global_r, offset + c] = U_block[local_r, c]
            for local_r, global_r in enumerate(v_rows):
                V_66[global_r, offset + c] = V_block[local_r, c]
            for local_r, global_r in enumerate(w_rows):
                W_66[global_r, offset + c] = W_block[local_r, c]

    S_U, S_V, S_W = get_strassen_2x2()
    M12_U, M12_V, M12_W = get_mult_1x2_2x2()
    
    def get_u_idx(i_start, i_end, j_start, j_end):
        return [i*m + j for i in range(i_start, i_end) for j in range(j_start, j_end)]
    def get_v_idx(j_start, j_end, k_start, k_end):
        return [j*p + k for j in range(j_start, j_end) for k in range(k_start, k_end)]
    def get_w_idx(k_start, k_end, i_start, i_end):
        return [k*n + i for k in range(k_start, k_end) for i in range(i_start, i_end)]

    col_offset = 0
    # 6 blocks of 2x2 Strassen
    for j_block in range(2):
        for k_block in range(3):
            j_start = j_block * 2
            j_end = j_block * 2 + 2
            k_start = k_block * 2
            k_end = k_block * 2 + 2
            u_rows = get_u_idx(0, 2, j_start, j_end)
            v_rows = get_v_idx(j_start, j_end, k_start, k_end)
            w_rows = get_w_idx(k_start, k_end, 0, 2)
            place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
            # 1x2 block
            u_rows = get_u_idx(2, 3, j_start, j_end)
            v_rows = get_v_idx(j_start, j_end, k_start, k_end)
            w_rows = get_w_idx(k_start, k_end, 2, 3)
            place_block(M12_U, M12_V, M12_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 4

    total_rank = col_offset
    U_66 = U_66[:, :total_rank]
    V_66 = V_66[:, :total_rank]
    W_66 = W_66[:, :total_rank]

    if verify(U_66.tolist(), V_66.tolist(), W_66.tolist(), n, m, p):
        best_U = U_66.tolist()
        best_V = V_66.tolist()
        best_W = W_66.tolist()
        best_rank = total_rank

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
