"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
        terms = []
        # p1 = (a11+a22)*(b11+b22)
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2 = (a21+a22)*b11
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3 = a11*(b12-b22)
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4 = a22*(b21-b11)
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5 = (a11+a12)*b22
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6 = (a21-a11)*(b11+b12)
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7 = (a12-a22)*(b21+b22)
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    # Build the full 3x4 * 4x4 decomposition.
    # Split strategy:
    #   Rows of A: top 2 (i=0,1), bottom 1 (i=2)
    #   Cols of B / rows of C: left 2 (k=0,1), right 2 (k=2,3)
    #   Middle dim j: j=0,1 and j=2,3
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  -> 7+7=14
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  -> 7+7=14
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]  -> 2+2=4
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]  -> 2+2=4
    # Total: 36 (Wait, 14+14+4+4 = 36? No, 14+14 is 28. 4+4 is 8. Total 36?
    # Let's re-verify the counts in the code below.
    # 4 rank-7 blocks = 28.
    # 4 rank-4 blocks = 16.
    # Total 44.
    # The comment "Total: 36" in the original code was incorrect or referred to something else.
    # 14 (top-left 2x2 out) + 14 (top-right 2x2 out) = 28 for top 2 rows.
    # 4 (bottom-left 1x2 out) + 4 (bottom-right 1x2 out) = 8 for bottom row.
    # Total 36?
    # Let's check add_rank4_block_1x2x2x2.
    # It adds 4 terms.
    # There are 4 such blocks. 4 * 4 = 16.
    # 28 + 16 = 44.
    # So the comment "Total: 36" was wrong. The code produces 44.

    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 32
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 32
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 32

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block."""
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                U[(a_row_offset + ai) * m + (j_offset + aj)][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                W[(c_col_offset + ck) * n + (c_row_offset + ci)][col] += val
        return col_start + 7

    def add_rank4_block_1x4x4(a_row, col_start):
        """Add a rank-4 1x4*4x4 -> 1x4 block (schoolbook for single row)."""
        for j in range(4):
            col = col_start + j
            U[a_row * m + j][col] = 1
            for k in range(4):
                V[j * p + k][col] = 1
                W[k * n + a_row][col] = 1
        return col_start + 4

    col = 0
    # Top 2 rows of A, all 4x4 of B: 4 rank-7 Strassen blocks (2x2x2 each)
    # j={0,1}, k={0,1}
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    # j={2,3}, k={0,1}
    col = add_rank7_block(0, 0, 2, 0, 0, col)
    # j={0,1}, k={2,3}
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    # j={2,3}, k={2,3}
    col = add_rank7_block(0, 2, 2, 0, 2, col)
    # Bottom row of A: rank-4 schoolbook for 1x4x4
    col = add_rank4_block_1x4x4(2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            a_r = sum(U_arr[i*m+j, r] * A_test[i,j] for i in range(n) for j in range(m))
            b_r = sum(V_arr[j*p+k, r] * B_test[j,k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += W_arr[k*n+i, r] * a_r * b_r

        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-9:
            print(f"Verification failed: max error = {max_err}")
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception as e:
        print(f"Verification error: {e}")
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
