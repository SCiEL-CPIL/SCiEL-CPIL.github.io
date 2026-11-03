import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # Diagonal (matching) entries must sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Off-diagonal must vanish
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    def to_halfint_list(M, total_rank):
        return [[int(round(M[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(M.shape[0])]

    # ---- Attempt rank-32 via [2,3] column split ----
    # C = A_left(2x2) @ B_top(2x5) + A_right(2x2) @ B_bottom(2x5)
    # Each 2x2x5 split as cols [2,3]: rank7 (2x2x2) + rank9 (2x2x3) = 16
    # Two blocks => 32
    total_rank_32 = 32
    U32 = np.zeros((nm, total_rank_32), dtype=np.float64)
    V32 = np.zeros((mp, total_rank_32), dtype=np.float64)
    W32 = np.zeros((pn, total_rank_32), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    U[(a_i_offset + i_local) * m + (a_j_offset + j_local), col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    V[(b_j_offset + j_local) * p + (b_k_offset + k_local), col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    W[(c_k_offset + k_local) * n + (c_i_offset + i_local), col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x3_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        # Candidate rank-9 decomposition for 2x2x3.
        # U: 4 entries (i*2+j), V: 6 entries (j*3+k), W: 6 entries (k*2+i)
        patterns = [
            # 9 rank-1 terms
            ([1, 0, 0, 1], [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0]),
            ([1, 0, 0, 0], [0, 1, 0, 0, -1, 0], [0, 1, 0, 0, 1, 0]),
            ([0, 0, 0, 1], [-1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 0, 0, 1], [-1, 1, 0, 0, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0]),
            ([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]),
            ([0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    U[(a_i_offset + i_local) * m + (a_j_offset + j_local), col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 3
                    k_local = idx % 3
                    V[(b_j_offset + j_local) * p + (b_k_offset + k_local), col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    k_local = idx // 2
                    i_local = idx % 2
                    W[(c_k_offset + k_local) * n + (c_i_offset + i_local), col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    # A_left (rows 0-1) * B_top (rows 0-1, cols 0-4): cols [0-1] then [2-4]
    col = add_2x2x2_block(U32, V32, W32, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x3_block(U32, V32, W32, 0, 0, 0, 2, 0, 2, col)
    # A_right (rows 0-1, cols 2-3) * B_bottom (rows 2-3, cols 0-4)
    col = add_2x2x2_block(U32, V32, W32, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x3_block(U32, V32, W32, 0, 2, 2, 2, 0, 2, col)
    assert col == total_rank_32, col

    if verify_decomposition(U32, V32, W32):
        return {
            "U": to_halfint_list(U32, total_rank_32),
            "V": to_halfint_list(V32, total_rank_32),
            "W": to_halfint_list(W32, total_rank_32),
        }

    # ---- Fallback: verified rank-36 solution ----
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    U[(a_i_offset + i_local) * m + (a_j_offset + j_local), col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    V[(b_j_offset + j_local) * p + (b_k_offset + k_local), col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    W[(c_k_offset + k_local) * n + (c_i_offset + i_local), col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    if verify_decomposition(U, V, W):
        return {
            "U": to_halfint_list(U, total_rank),
            "V": to_halfint_list(V, total_rank),
            "W": to_halfint_list(W, total_rank),
        }

    return best_result
# EVOLVE-BLOCK-END
