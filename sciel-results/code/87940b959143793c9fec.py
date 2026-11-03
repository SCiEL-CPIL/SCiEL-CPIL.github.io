import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

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

    # Strassen-based decomposition for 2x4x5
    # Split the middle dimension (size 4) into two groups of 2
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B = [B_L; B_R] where B_L, B_R are 2x5
    # C = A_L * B_L + A_R * B_R
    #
    # For each 2x2 * 2x5:
    # Split the 5 columns into 2+2+1
    # 2x2*2x2: rank 7 via Strassen
    # 2x2*2x2: rank 7 via Strassen
    # 2x2*2x1: rank 4 via schoolbook
    # Total per 2x2*2x5: 18
    # Total for 2x4x5: 18 + 18 = 36

    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        """
        Add a 2x2x2 block decomposition using Strassen's algorithm.
        A block is at rows [a_i_offset, a_i_offset+1], cols [a_j_offset, a_j_offset+1]
        B block is at rows [b_j_offset, b_j_offset+1], cols [b_k_offset, b_k_offset+1]
        C block is at rows [c_i_offset, c_i_offset+1], cols [c_k_offset, c_k_offset+1]
        """
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [-1, 1, 0, 0]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
        ]

        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val

            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val

            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val

            col_offset += 1

        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        """
        Add a 2x2x1 block decomposition (schoolbook).
        A block is 2x2, B block is 2x1, C block is 2x1.
        """
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
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val

            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val

            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val

            col_offset += 1

        return col_offset

    # Build the decomposition
    col = 0

    # A_left * B_top: A[:, 0:2] * B[0:2, :]
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)

    # A_right * B_bottom: A[:, 2:4] * B[2:4, :]
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"

    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])

        expected = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i * m + j, j * p + k, k * n + i] = 1.0

        if np.max(np.abs(T - expected)) > 1e-9:
            return False
        return True

    if time.time() > deadline:
        return best_result

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]

        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank
    else:
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = rank_school

    return best_result
# EVOLVE-BLOCK-END
