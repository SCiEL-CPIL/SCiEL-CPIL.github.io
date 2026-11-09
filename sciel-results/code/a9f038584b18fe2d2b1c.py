"""Rewritten program for matrix multiplication 4x5x6 rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS
    total_cells = n * m * p  # 120

    # Schoolbook fallback
    U_school = np.zeros((n * m, total_cells), dtype=object)
    V_school = np.zeros((m * p, total_cells), dtype=object)
    W_school = np.zeros((p * n, total_cells), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist(),
        "rank": total_cells
    }

    if time.time() + SAFETY_MARGIN > DEADLINE:
        return best_result

    # Strassen 2x2x2 decomposition
    strassen_u = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: -A00 + A10
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    strassen_v = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 0, 1, -1],  # M3: B10 - B11
        [-1, 1, 0, 0],  # M4: -B00 + B01
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    strassen_w = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, -1],  # M3: C01 - C11
        [-1, 0, 1, 0],  # M4: -C00 + C10
        [0, 1, 0, 1],   # M5: C01 + C11
        [1, 0, 0, 0],   # M6: C00
        [-1, 0, 0, 1]   # M7: -C00 + C11
    ]

    def verify(U, V, W, nn, mm, pp):
        rank = U.shape[1]
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    total = 0
                    for t in range(rank):
                        total += U[i * mm + j, t] * V[j * pp + k, t] * W[k * nn + i, t]
                    if total != 1:
                        return False
        return True

    def add_strassen_block(U, V, W, i0, j0, k0, col, nn, mm, pp):
        """Add a Strassen 2x2x2 block at global offset (i0,j0,k0). Returns new col."""
        for t in range(7):
            for ai in range(2):
                for aj in range(2):
                    local_idx = ai * 2 + aj
                    coeff = strassen_u[t][local_idx]
                    if coeff != 0:
                        U[(i0 + ai) * mm + (j0 + aj), col] = coeff
            for bj_l in range(2):
                for bk_l in range(2):
                    local_idx = bj_l * 2 + bk_l
                    coeff = strassen_v[t][local_idx]
                    if coeff != 0:
                        V[(j0 + bj_l) * pp + (k0 + bk_l), col] = coeff
            for ci in range(2):
                for ck in range(2):
                    local_idx = ci * 2 + ck
                    coeff = strassen_w[t][local_idx]
                    if coeff != 0:
                        W[(k0 + ck) * nn + (i0 + ci), col] = coeff
            col += 1
        return col

    def add_schoolbook_block(U, V, W, i0, j0, k0, ii, jj, kk, col, nn, mm, pp):
        """Add a schoolbook block of size ii x jj x kk at offset (i0,j0,k0). Returns new col."""
        for i in range(ii):
            for j in range(jj):
                for k in range(kk):
                    U[(i0 + i) * mm + (j0 + j), col] = 1
                    V[(j0 + j) * pp + (k0 + k), col] = 1
                    W[(k0 + k) * nn + (i0 + i), col] = 1
                    col += 1
        return col

    # Strategy: partition 4x5x6 as (2+2) x (3+2) x (3+3)
    # Eight blocks covering full k range [0,6):
    #   Block A: i=[0,2), j=[0,3), k=[0,3)  -> 2x3x3, rank 17
    #   Block B: i=[0,2), j=[3,5), k=[0,3)  -> 2x2x3, rank 11
    #   Block C: i=[2,4), j=[0,3), k=[0,3)  -> 2x3x3, rank 17
    #   Block D: i=[2,4), j=[3,5), k=[0,3)  -> 2x2x3, rank 11
    #   Block E: i=[0,2), j=[0,3), k=[3,6)  -> 2x3x3, rank 17
    #   Block F: i=[0,2), j=[3,5), k=[3,6)  -> 2x2x3, rank 11
    #   Block G: i=[2,4), j=[0,3), k=[3,6)  -> 2x3x3, rank 17
    #   Block H: i=[2,4), j=[3,5), k=[3,6)  -> 2x2x3, rank 11
    # Total rank: 4*17 + 4*11 = 68 + 44 = 112

    total_rank = 112

    if total_rank >= best_result["rank"]:
        return best_result

    U_new = np.zeros((n * m, total_rank), dtype=object)
    V_new = np.zeros((m * p, total_rank), dtype=object)
    W_new = np.zeros((p * n, total_rank), dtype=object)

    col = 0

    # Helper: decompose a 2x3x3 block (i_size=2, j_size=3, k_size=3)
    # Partition j as 2+1, k as 2+1:
    #   (2,2,2): Strassen rank 7
    #   (2,2,1): schoolbook rank 4
    #   (2,1,2): schoolbook rank 4
    #   (2,1,1): schoolbook rank 2
    # Total: 17
    def add_2x3x3(U, V, W, i0, j0, k0, col, nn, mm, pp):
        # Sub-block 1: (2,2,2) at (i0, j0, k0)
        col = add_strassen_block(U, V, W, i0, j0, k0, col, nn, mm, pp)
        # Sub-block 2: (2,2,1) at (i0, j0, k0+2)
        col = add_schoolbook_block(U, V, W, i0, j0, k0 + 2, 2, 2, 1, col, nn, mm, pp)
        # Sub-block 3: (2,1,2) at (i0, j0+2, k0)
        col = add_schoolbook_block(U, V, W, i0, j0 + 2, k0, 2, 1, 2, col, nn, mm, pp)
        # Sub-block 4: (2,1,1) at (i0, j0+2, k0+2)
        col = add_schoolbook_block(U, V, W, i0, j0 + 2, k0 + 2, 2, 1, 1, col, nn, mm, pp)
        return col

    # Helper: decompose a 2x2x3 block (i_size=2, j_size=2, k_size=3)
    # Partition k as 2+1:
    #   (2,2,2): Strassen rank 7
    #   (2,2,1): schoolbook rank 4
    # Total: 11
    def add_2x2x3(U, V, W, i0, j0, k0, col, nn, mm, pp):
        # Sub-block 1: (2,2,2) at (i0, j0, k0)
        col = add_strassen_block(U, V, W, i0, j0, k0, col, nn, mm, pp)
        # Sub-block 2: (2,2,1) at (i0, j0, k0+2)
        col = add_schoolbook_block(U, V, W, i0, j0, k0 + 2, 2, 2, 1, col, nn, mm, pp)
        return col

    # Block A: i=[0,2), j=[0,3), k=[0,3)
    col = add_2x3x3(U_new, V_new, W_new, 0, 0, 0, col, n, m, p)
    # Block B: i=[0,2), j=[3,5), k=[0,3)
    col = add_2x2x3(U_new, V_new, W_new, 0, 3, 0, col, n, m, p)
    # Block C: i=[2,4), j=[0,3), k=[0,3)
    col = add_2x3x3(U_new, V_new, W_new, 2, 0, 0, col, n, m, p)
    # Block D: i=[2,4), j=[3,5), k=[0,3)
    col = add_2x2x3(U_new, V_new, W_new, 2, 3, 0, col, n, m, p)
    # Block E: i=[0,2), j=[0,3), k=[3,6)
    col = add_2x3x3(U_new, V_new, W_new, 0, 0, 3, col, n, m, p)
    # Block F: i=[0,2), j=[3,5), k=[3,6)
    col = add_2x2x3(U_new, V_new, W_new, 0, 3, 3, col, n, m, p)
    # Block G: i=[2,4), j=[0,3), k=[3,6)
    col = add_2x3x3(U_new, V_new, W_new, 2, 0, 3, col, n, m, p)
    # Block H: i=[2,4), j=[3,5), k=[3,6)
    col = add_2x2x3(U_new, V_new, W_new, 2, 3, 3, col, n, m, p)

    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"

    if verify(U_new, V_new, W_new, n, m, p):
        best_result = {
            "U": U_new.tolist(),
            "V": V_new.tolist(),
            "W": W_new.tolist(),
            "rank": total_rank
        }
    else:
        # If verification fails, fall back to a simpler but correct approach
        # Use the previous 108-rank decomposition as fallback
        total_rank_fb = 108
        U_fb = np.zeros((n * m, total_rank_fb), dtype=object)
        V_fb = np.zeros((m * p, total_rank_fb), dtype=object)
        W_fb = np.zeros((p * n, total_rank_fb), dtype=object)
        col = 0
        for bi in range(2):
            for bj in range(2):
                for bk in range(3):
                    col = add_strassen_block(U_fb, V_fb, W_fb, 2*bi, 2*bj, 2*bk, col, n, m, p)
        # Remaining: j=4, all i and k
        col = add_schoolbook_block(U_fb, V_fb, W_fb, 0, 4, 0, 4, 1, 6, col, n, m, p)
        assert col == total_rank_fb
        if verify(U_fb, V_fb, W_fb, n, m, p):
            best_result = {
                "U": U_fb.tolist(),
                "V": V_fb.tolist(),
                "W": W_fb.tolist(),
                "rank": total_rank_fb
            }

    return best_result
# EVOLVE-BLOCK-END
