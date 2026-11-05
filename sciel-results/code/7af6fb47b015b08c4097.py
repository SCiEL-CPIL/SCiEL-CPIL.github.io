"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

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

    # Strategy: split middle dim j as 3+1
    # Block 1: A[0:3, 0:3] * B[0:3, 0:4] -> C[0:3, 0:4]
    #   Sub-blocks:
    #     C[0:2, 0:2] = A[0:2,0:3]*B[0:3,0:2] -> rank 7 (Strassen)
    #     C[0:2, 2:4] = A[0:2,0:3]*B[0:3,2:4] -> rank 7 (Strassen)
    #     C[2:3, 0:4] = A[2:3,0:3]*B[0:3,0:4] -> rank 3 (schoolbook)
    # Block 2: A[0:3, 3:4] * B[3:4, 0:4] -> C[0:3, 0:4]
    #   C[i,k] += A[i,3]*B[3,k] -> rank 3 (schoolbook)
    # Total: 7 + 7 + 3 + 3 = 20

    total_rank = 20
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 20
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 20
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 20

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block for C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]."""
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

    def add_rank3_block_1x3x4(a_row, b_col_offset, j_offset, col_start):
        """Add rank-3 schoolbook for A[a_row, j_offset:j_offset+3] * B[j_offset:j_offset+3, b_col_offset:b_col_offset+4].
        Actually this is 1x3 * 3x4 -> 1x4, which needs rank 4.
        Wait, let me reconsider.
        """
        # For C[a_row, 0:4] += A[a_row, j_offset:j_offset+3] * B[j_offset:j_offset+3, 0:4]
        # This is 1x3 * 3x4 = 1x4, needs rank 4 (not 3)
        pass

    # Let me reconsider the decomposition more carefully.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into {0,1,2} and {3}
    
    # Part 1: j in {0,1,2}
    #   For i in {0,1}: C[0:2, k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    #     Split k into {0,1} and {2,3}:
    #       C[0:2, 0:2] = A[0:2, 0:3] * B[0:3, 0:2] -> 2x3 * 3x2, rank 7
    #       C[0:2, 2:4] = A[0:2, 0:3] * B[0:3, 2:4] -> 2x3 * 3x2, rank 7
    #   For i=2: C[2, k] = sum_{j=0}^{2} A[2,j]*B[j,k]
    #     C[2, 0:4] = A[2, 0:3] * B[0:3, 0:4] -> 1x3 * 3x4, rank 4
    
    # Part 2: j=3
    #   For all i in {0,1,2}: C[i,k] += A[i,3]*B[3,k]
    #     C[0:3, 0:4] += A[0:3, 3] * B[3, 0:4] -> 3x1 * 1x4, rank 4
    
    # Total: 7 + 7 + 4 + 4 = 22

    total_rank = 22
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 22
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 22
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 22

    col = 0
    # C[0:2, 0:2] = A[0:2, 0:3] * B[0:3, 0:2] -> rank 7
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    # C[0:2, 2:4] = A[0:2, 0:3] * B[0:3, 2:4] -> rank 7
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    # C[2, 0:4] = A[2, 0:3] * B[0:3, 0:4] -> rank 4
    for r in range(4):
        j_idx = r % 3
        k_idx = r // 3
        if r < 3:
            j_idx = r
            k_idx = 0
        else:
            j_idx = 2
            k_idx = 1
        # Actually, 1x3 * 3x4 needs rank 4:
        # C[2,0] += A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        # This is a dot product, which has rank 3 (not 4)
        # Wait, 1x3 * 3x4: the output is 1x4, each element is a dot product of length 3
        # The tensor is sum_j A[2,j] * B[j,k], which has rank 3 (one term per j)
        pass

    # Let me just use schoolbook for the 1x3*3x4 part (rank 3) and 3x1*1x4 part (rank 3)
    # Total: 7 + 7 + 3 + 3 = 20

    total_rank = 20
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    # C[0:2, 0:2] = A[0:2, 0:3] * B[0:3, 0:2] -> rank 7
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    # C[0:2, 2:4] = A[0:2, 0:3] * B[0:3, 2:4] -> rank 7
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    # C[2, 0:4] = A[2, 0:3] * B[0:3, 0:4] -> rank 3
    for j_idx in range(3):
        col_idx = col
        U[2 * m + j_idx][col_idx] = 1
        V[j_idx * p + 0][col_idx] = 1
        W[0 * n + 2][col_idx] = 1
        V[j_idx * p + 1][col_idx] = 1
        W[1 * n + 2][col_idx] = 1
        V[j_idx * p + 2][col_idx] = 1
        W[2 * n + 2][col_idx] = 1
        V[j_idx * p + 3][col_idx] = 1
        W[3 * n + 2][col_idx] = 1
        col += 1
    # C[0:3, 0:4] += A[0:3, 3] * B[3, 0:4] -> rank 3
    for i_idx in range(3):
        col_idx = col
        U[i_idx * m + 3][col_idx] = 1
        V[3 * p + 0][col_idx] = 1
        W[0 * n + i_idx][col_idx] = 1
        V[3 * p + 1][col_idx] = 1
        W[1 * n + i_idx][col_idx] = 1
        V[3 * p + 2][col_idx] = 1
        W[2 * n + i_idx][col_idx] = 1
        V[3 * p + 3][col_idx] = 1
        W[3 * n + i_idx][col_idx] = 1
        col += 1

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition
    try:
        import numpy as np
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
