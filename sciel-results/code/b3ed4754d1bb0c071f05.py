import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START_TIME

def construct():
    n, m, p = 4, 4, 9

    # Known optimal rank for 2x2x2 is 7 (Strassen)
    # We'll build the 4x4x4x4 decomposition using a 2x2 block Strassen
    # and the 4x4x4x1 using schoolbook (rank 16, which is optimal)

    # First, define the Strassen 2x2x2 decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Where A is 2x2, B is 2x2, C = A*B is 2x2
    # Indices: A[0,0], A[0,1], A[1,0], A[1,1]
    #           B[0,0], B[0,1], B[1,0], B[1,1]
    #           C[0,0], C[0,1], C[1,0], C[1,1]

    # The 7 terms of Strassen:
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

    # For each term r in 0..6:
    # U[:,r], V[:,r], W[:,r]

    strassen_2x2_U = []
    strassen_2x2_V = []
    strassen_2x2_W = []

    def add_strassen_2x2(u_dict, v_dict, w_dict):
        # u_dict: mapping (i,j) -> coeff
        # v_dict: mapping (j,k) -> coeff
        # w_dict: mapping (k,i) -> coeff
        u_vec = [0.0] * 4
        v_vec = [0.0] * 4
        w_vec = [0.0] * 4
        for (i,j), c in u_dict.items():
            u_vec[i*2+j] += c
        for (j,k), c in v_dict.items():
            v_vec[j*2+k] += c
        for (k,i), c in w_dict.items():
            w_vec[k*2+i] += c
        strassen_2x2_U.append(u_vec)
        strassen_2x2_V.append(v_vec)
        strassen_2x2_W.append(w_vec)

    # M1 = (A00+A11)*(B00+B11) -> contributes to C00(+1), C11(+1)
    add_strassen_2x2({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2 = (A10+A11)*B00 -> contributes to C10(+1), C11(-1)
    add_strassen_2x2({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):-1, (1,0):1})
    # M3 = A00*(B01-B11) -> contributes to C01(+1), C11(+1)
    add_strassen_2x2({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})
    # M4 = A11*(B10-B00) -> contributes to C00(+1), C10(+1)
    add_strassen_2x2({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})
    # M5 = (A00+A01)*B11 -> contributes to C00(-1), C01(+1)
    add_strassen_2x2({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11(+1)
    add_strassen_2x2({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00(+1)
    add_strassen_2x2({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})

    # Now build 4x4x4x4 decomposition using 2x2 block Strassen
    # A is 4x4, split into 2x2 blocks of 2x2 submatrices
    # B is 4x4, same
    # C = A*B is 4x4

    # Block indices: block (br, bc) has rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11

    # Strassen at block level:
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

    # Each Mi is a 2x2 * 2x2 product, decomposed with Strassen (rank 7)
    # So total rank for 4x4x4x4 is 7*7 = 49

    U_4x4_4x4 = []
    V_4x4_4x4 = []
    W_4x4_4x4 = []

    # For each block-level Strassen term (block_row_A, block_col_A, coeff_A) etc.
    # We need to map to the 4x4 indices

    # Let's define the 7 block-level terms with their A, B, C block coefficients
    # Each term: (list of (block_i, block_j, coeff) for A, list of (block_i, block_j, coeff) for B, list of (block_i, block_j, coeff) for C)

    block_terms = [
        # M1 = (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10+A11)*B00 -> C10(+1), C11(-1)
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A00*(B01-B11) -> C01(+1), C11(+1)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A11*(B10-B00) -> C00(+1), C10(+1)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A00+A01)*B11 -> C00(-1), C01(+1)
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A10-A00)*(B00+B01) -> C11(+1)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)*(B10+B11) -> C00(+1)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    for A_blocks, B_blocks, C_blocks in block_terms:
        for r in range(7):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 16
            w_vec = [0.0] * 16

            # u_vec: A is 4x4, index i*4+j
            # For each (block_i, block_j, coeff) in A_blocks:
            #   For each (local_i, local_j) in 2x2:
            #     u_vec[(2*block_i+local_i)*4 + (2*block_j+local_j)] += coeff * strassen_2x2_U[r][local_i*2+local_j]
            for (bi, bj, coeff) in A_blocks:
                for li in range(2):
                    for lj in range(2):
                        u_vec[(2*bi+li)*4 + (2*bj+lj)] += coeff * strassen_2x2_U[r][li*2+lj]

            # v_vec: B is 4x4, index j*4+k
            for (bi, bj, coeff) in B_blocks:
                for lj in range(2):
                    for lk in range(2):
                        v_vec[(2*bi+lj)*4 + (2*bj+lk)] += coeff * strassen_2x2_V[r][lj*2+lk]

            # w_vec: C is 4x4, index k*4+i (note: k is row, i is col in C)
            for (bi, bj, coeff) in C_blocks:
                for lk in range(2):
                    for li in range(2):
                        w_vec[(2*bi+lk)*4 + (2*bj+li)] += coeff * strassen_2x2_W[r][lk*2+li]

            U_4x4_4x4.append(u_vec)
            V_4x4_4x4.append(v_vec)
            W_4x4_4x4.append(w_vec)

    # Now build the full 4x4x4x9 decomposition
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # A*B1: 4x4*4x4, rank 49
    # A*B2: 4x4*4x4, rank 49
    # A*B3: 4x4*4x1, rank 16 (schoolbook)
    # Total: 49 + 49 + 16 = 114

    U_full = []
    V_full = []
    W_full = []

    # Part 1: A * B1 (cols 0-3)
    # B1 is B[:, 0:4]
    # For each term in U_4x4_4x4, V_4x4_4x4, W_4x4_4x4:
    #   u is for A (4x4), v is for B1 (4x4), w is for C1 (4x4)
    #   Need to extend v to 4x9 (cols 4-8 are 0), w to 4x9 (cols 4-8 are 0)
    for r in range(49):
        u_vec = U_4x4_4x4[r]
        v_vec_4x4 = V_4x4_4x4[r]
        w_vec_4x4 = W_4x4_4x4[r]

        v_vec_full = [0.0] * 36  # 4x9
        w_vec_full = [0.0] * 36  # 4x9, but C is 4x9, index k*4+i where k is 0..8, i is 0..3

        # v_vec: B is 4x9, index j*9+k
        # B1 occupies cols 0-3, so for each j in 0..3, k in 0..3:
        for j in range(4):
            for k in range(4):
                v_vec_full[j*9+k] = v_vec_4x4[j*4+k]

        # w_vec: C is 4x9, index k*4+i (k is row 0..8, i is col 0..3)
        # C1 occupies rows 0-3, so for each k in 0..3, i in 0..3:
        for k in range(4):
            for i in range(4):
                w_vec_full[k*4+i] = w_vec_4x4[k*4+i]

        U_full.append(u_vec)
        V_full.append(v_vec_full)
        W_full.append(w_vec_full)

    # Part 2: A * B2 (cols 4-7)
    # B2 is B[:, 4:8]
    for r in range(49):
        u_vec = U_4x4_4x4[r]
        v_vec_4x4 = V_4x4_4x4[r]
        w_vec_4x4 = W_4x4_4x4[r]

        v_vec_full = [0.0] * 36
        w_vec_full = [0.0] * 36

        # v_vec: B2 occupies cols 4-7
        for j in range(4):
            for k in range(4):
                v_vec_full[j*9 + (k+4)] = v_vec_4x4[j*4+k]

        # w_vec: C2 occupies rows 4-7
        for k in range(4):
            for i in range(4):
                w_vec_full[(k+4)*4 + i] = w_vec_4x4[k*4+i]

        U_full.append(u_vec)
        V_full.append(v_vec_full)
        W_full.append(w_vec_full)

    # Part 3: A * B3 (col 8) - schoolbook: 16 terms
    # A is 4x4, B3 is 4x1, C3 is 4x1
    # C3[i,8] = sum_j A[i,j] * B3[j,8]
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 36
            w_vec = [0.0] * 36
            u_vec[i*4+j] = 1.0
            v_vec[j*9+8] = 1.0
            w_vec[8*4+i] = 1.0
            U_full.append(u_vec)
            V_full.append(v_vec)
            W_full.append(w_vec)

    rank = len(U_full)

    # Convert to numpy arrays
    U = np.array(U_full, dtype=float)
    V = np.array(V_full, dtype=float)
    W = np.array(W_full, dtype=float)

    # Verify with random matrices
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_test = A_test @ B_test

    C_decomp = np.zeros((4, 9))
    for r in range(rank):
        u_mat = U[:, r].reshape(4, 4)
        v_mat = V[:, r].reshape(4, 9)
        w_mat = W[:, r].reshape(9, 4)
        C_decomp += np.einsum('ij,jk,ki->ik', u_mat, v_mat, w_mat)

    error = np.max(np.abs(C_decomp - C_test))

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
