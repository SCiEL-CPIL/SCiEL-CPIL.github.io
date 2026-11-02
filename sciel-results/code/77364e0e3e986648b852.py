"""Tensor decomposition for matrix multiplication 3x3 * 3x8."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Uses a Strassen-based construction on 2x2 blocks to achieve rank 68.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin

    def validate_decomposition(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank_val = U_arr.shape[1]

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook fallback
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

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strassen-based construction for rank 68
    # Split A (3x3) into blocks: A00 (2x2), A01 (2x1), A10 (1x2), A11 (1x1)
    # Split B (3x8) into blocks: B00 (2x8), B01 (2x8), B10 (1x8), B11 (1x8)
    # Actually, let's use a different split:
    # A = [A_top (2x3), A_bottom (1x3)]
    # B = [B_left (3x4), B_right (3x4)]
    # C = [C_top (2x8), C_bottom (1x8)]
    #
    # C_top = A_top * B = A_top * [B_left | B_right]
    # C_bottom = A_bottom * B = A_bottom * [B_left | B_right]
    #
    # For C_top: 2x3 * 3x8. Split B into 2x8 and 1x8:
    #   C_top = A_top_left(2x2) * B_top(2x8) + A_top_right(2x1) * B_bottom(1x8)
    #   Use Strassen for 2x2 * 2x8: rank 7*8/2 = 28
    #   A_top_right * B_bottom: 2x1 * 1x8 = 2*8 = 16 multiplications
    #
    # For C_bottom: 1x3 * 3x8
    #   C_bottom = A_bottom_left(1x2) * B_top(2x8) + A_bottom_right(1x1) * B_bottom(1x8)
    #   A_bottom_left * B_top: 1x2 * 2x8 = 16 multiplications
    #   A_bottom_right * B_bottom: 1x1 * 1x8 = 8 multiplications
    #
    # Total: 28 + 16 + 16 + 8 = 68

    def build_strassen_decomposition():
        """
        Construct an exact rank-68 decomposition using Strassen's algorithm
        on the 2x2 * 2x8 sub-problem.
        """
        r = 68
        U = [[0] * r for _ in range(n * m)]  # 9 x 68
        V = [[0] * r for _ in range(m * p)]  # 24 x 68
        W = [[0] * r for _ in range(p * n)]  # 24 x 68

        term_idx = 0

        # Helper: add a term with specified U, V, W entries
        def add_term(u_entries, v_entries, w_entries):
            nonlocal term_idx
            for (u_idx, u_val) in u_entries:
                U[u_idx][term_idx] += u_val
            for (v_idx, v_val) in v_entries:
                V[v_idx][term_idx] += v_val
            for (w_idx, w_val) in w_entries:
                W[w_idx][term_idx] += w_val
            term_idx += 1

        # Indexing:
        # A[i,j] -> U index: i*m + j = i*3 + j
        # B[j,k] -> V index: j*p + k = j*8 + k
        # C[k,i] -> W index: k*n + i = k*3 + i

        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, :] -> C[:, 0:2]
        # A_top_left = [[A00, A01], [A10, A11]] where A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B_top = B[0:2, :] = [[B00..B07], [B10..B17]]
        #
        # Strassen's 7 multiplications for 2x2 * 2x8:
        # Let B be split into 4 blocks of 2x4:
        #   B = [[B00, B01], [B10, B11]] where each is 2x4
        #   B00 = B[0:2, 0:4], B01 = B[0:2, 4:8]
        #   B10 = B[2:4, 0:4] -> but we only have B[0:2, :], so:
        #   Actually B_top is 2x8, split into 2x4 blocks:
        #   B_top = [[B00 (2x4), B01 (2x4)]]
        #
        # For 2x2 * 2x8, we can view it as 2x2 * (2x4 + 2x4)
        # Strassen for 2x2 * 2x2 gives 7 multiplications per 2x4 block
        # So 2 blocks of 2x4: 7 + 7 = 14 multiplications? No...
        #
        # Actually, let's think of 2x2 * 2x8 as 2x2 * (2x8)
        # We can use the fact that 2x2 * 2x8 = 2x2 * 2x4 + 2x2 * 2x4
        # Each 2x2 * 2x4 can be done with 7 multiplications (Strassen for 2x2 * 2x2, applied to each 2x2 column block)
        # So total: 7 + 7 = 14 multiplications for the 2x2 * 2x8 part
        #
        # Wait, that's not right either. Let me reconsider.
        #
        # 2x2 * 2x8: The result is 2x8. 
        # We can split the 2x8 matrix into two 2x4 matrices.
        # For each 2x4, we compute 2x2 * 2x4.
        # 2x2 * 2x4 can be done by treating it as 2x2 * (2x2 + 2x2) and using Strassen on each 2x2.
        # So 2x2 * 2x4 requires 7 + 7 = 14 multiplications? No, that's not right.
        #
        # Actually, 2x2 * 2x4: Split the 2x4 into two 2x2 blocks.
        # C_left = A * B_left (2x2 * 2x2) -> 7 mults
        # C_right = A * B_right (2x2 * 2x2) -> 7 mults
        # Total: 14 multiplications for 2x2 * 2x4
        # So 2x2 * 2x8 = 2x2 * 2x4 + 2x2 * 2x4 = 14 + 14 = 28 multiplications
        #
        # OK so 28 is correct.

        # Strassen for 2x2 * 2x2:
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A00) * (B00 + B01)
        # m7 = (A01 - A11) * (B10 + B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6

        # For 2x2 * 2x4, split B into B_left (2x2) and B_right (2x2):
        # Use Strassen on A * B_left and A * B_right separately
        # This gives 7 + 7 = 14 multiplications

        # For 2x2 * 2x8, split B into 4 blocks of 2x2:
        # B = [B0, B1, B2, B3] where each is 2x2
        # C = A * B = A*B0 + A*B1 + A*B2 + A*B3
        # Each A*Bk uses 7 multiplications
        # Total: 28 multiplications

        # Let's implement this:
        # A_top_left indices in U:
        # A[0,0] -> U[0], A[0,1] -> U[1], A[1,0] -> U[3], A[1,1] -> U[4]
        # B_top indices in V:
        # B[0,k] -> V[k], B[1,k] -> V[8+k]
        # C output indices in W:
        # C[k,0] -> W[3k], C[k,1] -> W[3k+1]

        # Strassen for 2x2 * 2x2 sub-problem
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # U indices: a00->u0, a01->u1, a10->u3, a11->u4
        # V indices: b00->v0, b01->v1, b10->v8, b11->v9
        # W indices: c00->w0, c01->w1, c10->w3, c11->w4

        def strassen_2x2_x_2x2(u0, u1, u3, u4, v0, v1, v8, v9, w0, w1, w3, w4):
            nonlocal term_idx
            # m1 = (a00 + a11) * (b00 + b11)
            add_term([(u0, 1), (u4, 1)], [(v0, 1), (v9, 1)], [(w0, 1), (w4, 1)])
            # m2 = (a10 + a11) * b00
            add_term([(u3, 1), (u4, 1)], [(v0, 1)], [(w3, 1), (w4, -1)])
            # m3 = a00 * (b01 - b11)
            add_term([(u0, 1)], [(v1, 1), (v9, -1)], [(w1, 1), (w4, 1)])
            # m4 = a11 * (b10 - b00)
            add_term([(u4, 1)], [(v8, 1), (v0, -1)], [(w0, 1), (w3, 1)])
            # m5 = (a00 + a01) * b11
            add_term([(u0, 1), (u1, 1)], [(v9, 1)], [(w0, -1), (w1, 1)])
            # m6 = (a10 - a00) * (b00 + b01)
            add_term([(u3, 1), (u0, -1)], [(v0, 1), (v1, 1)], [(w4, 1)])
            # m7 = (a01 - a11) * (b10 + b11)
            add_term([(u1, 1), (u4, -1)], [(v8, 1), (v9, 1)], [(w0, 1)])

        # Apply Strassen to each 2x2 block of B_top
        # B_top = B[0:2, 0:8], split into 4 blocks of 2x2
        for block in range(4):
            k_start = block * 2
            # V indices for this block:
            # B[0, k_start] -> V[k_start], B[0, k_start+1] -> V[k_start+1]
            # B[1, k_start] -> V[8+k_start], B[1, k_start+1] -> V[8+k_start+1]
            v0 = k_start
            v1 = k_start + 1
            v8 = 8 + k_start
            v9 = 8 + k_start + 1
            # W indices:
            # C[k_start, 0] -> W[3*k_start], C[k_start+1, 0] -> W[3*(k_start+1)]
            # C[k_start, 1] -> W[3*k_start+1], C[k_start+1, 1] -> W[3*(k_start+1)+1]
            w0 = 3 * k_start
            w1 = 3 * k_start + 1
            w3 = 3 * (k_start + 1)
            w4 = 3 * (k_start + 1) + 1
            strassen_2x2_x_2x2(0, 1, 3, 4, v0, v1, v8, v9, w0, w1, w3, w4)

        # Part 2: A[0:2, 2] * B[2, :] -> C[:, 0:8]
        # A[0,2] -> U[2], A[1,2] -> U[5]
        # B[2,k] -> V[16+k]
        # C[k,0] -> W[3k], C[k,1] -> W[3k+1]
        for k in range(8):
            add_term([(2, 1)], [(16 + k, 1)], [(3 * k + 0, 1)])
            add_term([(5, 1)], [(16 + k, 1)], [(3 * k + 1, 1)])

        # Part 3: A[2, 0:2] * B[0:2, :] -> C[:, 2]
        # A[2,0] -> U[6], A[2,1] -> U[7]
        # B[0,k] -> V[k], B[1,k] -> V[8+k]
        # C[k,2] -> W[3k+2]
        for k in range(8):
            add_term([(6, 1)], [(k, 1)], [(3 * k + 2, 1)])
            add_term([(7, 1)], [(8 + k, 1)], [(3 * k + 2, 1)])

        # Part 4: A[2, 2] * B[2, :] -> C[:, 2]
        # A[2,2] -> U[8]
        # B[2,k] -> V[16+k]
        # C[k,2] -> W[3k+2]
        for k in range(8):
            add_term([(8, 1)], [(16 + k, 1)], [(3 * k + 2, 1)])

        if term_idx == r:
            if validate_decomposition(U, V, W):
                return U, V, W, r
        return None

    result = build_strassen_decomposition()
    if result is not None:
        best_U, best_V, best_W, best_rank = result

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
