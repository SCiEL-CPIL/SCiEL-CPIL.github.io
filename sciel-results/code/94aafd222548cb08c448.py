import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Patterns: (A_coeffs, B_coeffs, C_coeffs)
    laderman_23_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,0):1,(2,0):-1}),
        ({(1,1):1}, {(2,1):1}, {(1,0):-1}),
        ({(2,1):1}, {(2,1):1}, {(2,0):1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,0):-1,(2,0):1}),
        ({(1,2):1}, {(2,2):1}, {(1,0):1}),
        ({(2,2):1}, {(2,2):1}, {(2,0):-1}),
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        ({(1,1):1}, {(2,2):1}, {(1,2):-1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):-1}),
        ({(2,1):1}, {(2,2):1}, {(2,2):-1}),
        ({(2,2):1}, {(2,1):1}, {(2,1):-1}),
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):1,(2,2):1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):1,(2,1):1}),
    ]

    # Strassen's 2x2 matrix multiplication algorithm (rank 7)
    # Patterns: (A_coeffs, B_coeffs, C_coeffs)
    # A, B, C are 2x2 matrices.
    strassen_7_patterns = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def build_hybrid_6x6():
        """
        Build 6x6x6 decomposition using Strassen on 2x2 blocks of 3x3 matrices.
        Rank = 7 (Strassen) * 23 (Laderman) = 161.
        """
        rank = 7 * 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # The 6x6 matrix is divided into 2x2 blocks of 3x3 submatrices.
        # Block indices: I, J, K in {0, 1}
        # Sub-block indices: pi, pj in {0, 1, 2}
        
        col_idx = 0
        for s_idx, (s_u, s_v, s_w) in enumerate(strassen_7_patterns):
            for l_idx, (l_u, l_v, l_w) in enumerate(laderman_23_patterns):
                # For each Strassen term, we have a Laderman decomposition of the 3x3 product.
                
                # U coefficients:
                # Strassen term s_u defines which 3x3 blocks of A are involved and their signs.
                # Laderman term l_u defines which elements within those 3x3 blocks are involved.
                # U[row, col] where row = (I * 3 + pi) * 6 + (J * 3 + pj)
                # Wait, A is 6x6.
                # A[I*3 + pi, J*3 + pj]
                
                for (si, sj), val_s in s_u.items():
                    # si, sj are block indices (0 or 1)
                    # This block corresponds to rows si*3..si*3+2 and cols sj*3..sj*3+2
                    for (pi, pj), val_l in l_u.items():
                        # pi, pj are local indices in 3x3 block
                        r = si * 3 + pi
                        c = sj * 3 + pj
                        U[r * m + c, col_idx] += val_s * val_l

                for (si, sj), val_s in s_v.items():
                    for (pi, pj), val_l in l_v.items():
                        r = si * 3 + pi
                        c = sj * 3 + pj
                        V[r * p + c, col_idx] += val_s * val_l

                for (si, sj), val_s in s_w.items():
                    for (pi, pj), val_l in l_w.items():
                        i = si * 3 + pi
                        k = sj * 3 + pj
                        W[k * n + i, col_idx] += val_s * val_l

                col_idx += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_test.flatten())
            M_v = np.dot(V_arr[:, r], B_test.flatten())
            M_r = M_u * M_v
            C_computed += M_r * W_arr[:, r].reshape(p, n).T
        error = np.max(np.abs(C_computed - C_expected))
        return error

    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    best_solution = None
    best_rank = float('inf')

    # Try Hybrid (Strassen 2x2 blocks of Laderman 3x3) -> Rank 161
    if time.time() < deadline:
        hybrid_sol, hybrid_rank = build_hybrid_6x6()
        U_h = np.array(hybrid_sol["U"], dtype=float)
        V_h = np.array(hybrid_sol["V"], dtype=float)
        W_h = np.array(hybrid_sol["W"], dtype=float)
        err_h = verify_decomposition(U_h, V_h, W_h, A_test, B_test, C_expected)
        if err_h < 1e-10:
            best_solution = hybrid_sol
            best_rank = hybrid_rank

    # Fallback to previous best (Laderman 2x2 blocks of 3x3) -> Rank 184
    if best_solution is None or time.time() < deadline - 10:
        # Re-implementing the 184 rank solution as fallback
        def build_laderman_6x6():
            rank = 8 * 23
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col_idx = 0
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                            for (pi, pj), val in a_coeffs.items():
                                r = I * 3 + pi
                                c = J * 3 + pj
                                U[r * m + c, col_idx] += val
                            for (pi, pj), val in b_coeffs.items():
                                r = J * 3 + pi
                                c = K * 3 + pj
                                V[r * p + c, col_idx] += val
                            for (pi, pj), val in c_coeffs.items():
                                i = I * 3 + pi
                                k = K * 3 + pj
                                W[k * n + i, col_idx] += val
                            col_idx += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # Fallback to schoolbook
    if best_solution is None:
        rank_trivial = n * m * p
        U_t = np.zeros((n * m, rank_trivial))
        V_t = np.zeros((m * p, rank_trivial))
        W_t = np.zeros((p * n, rank_trivial))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[i * m + j, idx] = 1
                    V_t[j * p + k, idx] = 1
                    W_t[k * n + i, idx] = 1
                    idx += 1
        best_solution = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    return best_solution
# EVOLVE-BLOCK-END
