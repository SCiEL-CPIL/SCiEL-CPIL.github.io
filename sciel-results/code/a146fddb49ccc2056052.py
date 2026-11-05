import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s limit

    # Strategy: Use Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = rank 65
    # But try to find a better decomposition for the 4x4x5 part.
    # 
    # Alternative: Treat 4x4x5 as (4x4x4) + (4x4x1) but see if we can do better
    # by exploiting the rectangular structure.
    #
    # Actually, let's try a different approach: use the fact that we can decompose
    # the 4x4x5 tensor using Strassen on the 4x4x4 block and then handle the 5th column.
    #
    # For the 4x4x4 part, we use Strassen (rank 49).
    # For the 4x4x1 part (computing C[:,4] = A @ B[:,4]), this is 4 dot products of length 4.
    # Each dot product has rank 4, so total rank 16 for this part.
    # Total: 49 + 16 = 65.
    #
    # Can we do better? Let's think about whether the 5th column can be integrated
    # into the Strassen decomposition more efficiently.
    #
    # Actually, let's try a completely different approach: use the Winograd F(4) transform
    # which gives rank 40 for 4x4x4, plus 16 for the 4x4x1 part = rank 56.
    #
    # The Winograd F(4) transform for 4x4x4 matrix multiplication:
    # This is based on the fact that 4 = 2*2, and we can use the tensor product of
    # two Strassen transforms, but with optimizations.
    #
    # Actually, the standard Winograd F(4) for matrix multiplication uses 40 multiplications.
    # Let me implement this carefully.
    #
    # The F(4) transform matrix (16x4) for rows of A:
    # Based on the Winograd paper, the transform is:
    # F_row[i, j] where i is the transformed index (0..15) and j is the original row (0..3)
    #
    # Let me use a known correct rank-40 decomposition for 4x4x4.
    #
    # Actually, I'll use a safer approach: implement the 4x4x4 using two levels of Strassen
    # (which gives rank 49) and see if I can verify it, then add the 4x4x1 part.
    #
    # Let me implement the Strassen 4x4x4 decomposition carefully.

    rank = 65
    U = np.zeros((n * m, rank), dtype=float)  # 16 x 65
    V = np.zeros((m * p, rank), dtype=float)  # 20 x 65
    W = np.zeros((p * n, rank), dtype=float)  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j, col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k, col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i, col] += c

    col = 0

    # Strassen 2x2x2 decomposition (7 multiplications)
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    inner_defs = [
        # (u_indices, v_indices, w_contribs)
        # u_indices: list of (local_row, local_col, coeff)
        # v_indices: list of (local_row, local_col, coeff)
        # w_contribs: list of (local_row, local_col, coeff)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),       # M1
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),                # M2
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),                # M3
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),                # M4
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),                # M5
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),                # M6
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),                # M7
    ]

    # For 4x4x4, we divide into 2x2 blocks.
    # A is 4x4, B is 4x4, C is 4x4.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bi is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2

    # Super-level Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    super_defs = [
        # (p_coeffs, q_coeffs, c_contribs)
        # p_coeffs: 4-element list for A blocks [A00, A01, A10, A11]
        # q_coeffs: 4-element list for B blocks [B00, B01, B10, B11]
        # c_contribs: list of (block_row, block_col, coeff)
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),       # M1 -> C00, C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),      # M2 -> C10, -C11
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),      # M3 -> C01, C11
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),      # M4 -> C00, C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),      # M5 -> -C00, C01
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),                 # M6 -> C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),                 # M7 -> C00
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook for C[:,4] = A @ B[:,4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1

    # Verify correctness with a random test
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 5)
    C_expected = A_test @ B_test

    C_computed = np.zeros((4, 5))
    for t in range(rank):
        u_vec = U[:, t]
        v_vec = V[:, t]
        w_vec = W[:, t]
        # u_vec is 16x1, v_vec is 20x1, w_vec is 20x1
        # C += outer(u_vec reshaped, v_vec reshaped) * w_vec
        u_mat = u_vec.reshape(4, 4)
        v_mat = v_vec.reshape(4, 5)
        w_mat = w_vec.reshape(5, 4)
        C_computed += (u_mat @ v_mat) * w_mat.T

    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-10:
        # Fallback to schoolbook if Strassen implementation is wrong
        rank = 80
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i * 4 + j, col] = 1
                    V[j * 5 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
