import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    laderman_23_patterns = [
        ({(0,0):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1,(2,0):1}, {(1,0):1}),
        ({(2,0):1}, {(2,0):1}, {(2,0):1}),
        ({(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1,(2,1):1}),
        ({(1,1):1}, {(2,1):1}, {(1,1):-1}),
        ({(2,1):1}, {(2,1):1}, {(2,1):-1}),
        ({(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1,(2,2):1}),
        ({(1,2):1}, {(2,2):1}, {(1,2):-1}),
        ({(2,2):1}, {(2,2):1}, {(2,2):-1}),
        ({(0,1):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1}),
        ({(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1}),
        ({(1,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(1,1):1}),
        ({(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(1,2):1}),
        ({(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}, {(2,1):1}),
        ({(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(2,2):1}),
        ({(0,0):1}, {(1,1):1,(2,1):1}, {(0,1):1}),
        ({(0,0):1}, {(1,2):1,(2,2):1}, {(0,2):1}),
        ({(1,1):1}, {(2,2):1}, {(1,2):1}),
        ({(1,2):1}, {(2,1):1}, {(1,1):1}),
        ({(2,1):1}, {(2,2):1}, {(2,2):1}),
        ({(2,2):1}, {(2,1):1}, {(2,1):1}),
        ({(1,1):1,(2,2):1}, {(0,1):1,(1,2):1}, {(1,1):-1,(2,2):-1}),
        ({(1,2):1,(2,1):1}, {(0,2):1,(1,1):1}, {(1,2):-1,(2,1):-1}),
    ]

    # Strassen 2x2 patterns
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def verify_3x3_patterns(patterns):
        np.random.seed(123)
        for trial in range(5):
            A_t = np.random.randn(3, 3)
            B_t = np.random.randn(3, 3)
            C_expected = A_t @ B_t
            C_computed = np.zeros((3, 3))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

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

    def build_laderman_6x6():
        """Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 2x2 blocks. Rank = 8*23 = 184."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for a_coeffs, b_coeffs, c_coeffs in laderman_23_patterns:
                        for (pi, pj), val in a_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in b_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in c_coeffs.items():
                            i = I * block_size + pi
                            k = K * block_size + pj
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_strassen_6x6():
        """Build 6x6x6 using Strassen on 3x3 blocks. Rank = 27*7 = 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_schoolbook_6x6():
        """Schoolbook rank 216."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def build_kronecker_6x6():
        """
        Build 6x6x6 decomposition using Kronecker product M_2 (Strassen) x M_3 (Laderman).
        Rank = 7 * 23 = 161.
        """
        rank = 7 * 23
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen coefficients for X_s = sum beta[s, I, J] A[I, J]
        # s is 0..6
        # I, J are block indices 0..1
        beta = np.zeros((7, 2, 2))
        beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
        beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
        beta[2, 0, 0] = 1
        beta[3, 1, 1] = 1
        beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
        beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
        beta[6, 0, 1] = 1; beta[6, 1, 1] = -1

        # Strassen coefficients for Y_s = sum gamma[s, J, K] B[J, K]
        gamma = np.zeros((7, 2, 2))
        gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
        gamma[1, 0, 0] = 1
        gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
        gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
        gamma[4, 1, 1] = 1
        gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
        gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1

        # Output coefficients alpha[s, I, K] for C[I, K] = sum alpha[s, I, K] Z_s
        alpha = np.zeros((7, 2, 2))
        # C11 = Z1 + Z4 - Z5 + Z7
        alpha[0, 0, 0] = 1; alpha[3, 0, 0] = 1; alpha[4, 0, 0] = -1; alpha[6, 0, 0] = 1
        # C12 = Z3 + Z5
        alpha[2, 0, 1] = 1; alpha[4, 0, 1] = 1
        # C21 = Z2 + Z4
        alpha[1, 1, 0] = 1; alpha[3, 1, 0] = 1
        # C22 = Z1 - Z2 + Z3 + Z6
        alpha[0, 1, 1] = 1; alpha[1, 1, 1] = -1; alpha[2, 1, 1] = 1; alpha[5, 1, 1] = 1

        # Laderman patterns are stored as lists of dicts
        # We need to convert them to arrays for faster access or just iterate
        # laderman_23_patterns is a list of 23 tuples of dicts
        
        for s in range(7):
            for r_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(laderman_23_patterns):
                col_idx = s * 23 + r_idx
                
                # Fill U
                # U[row, col] where row = (i_blk * 3 + i_loc) * 6 + (j_blk * 3 + j_loc)
                # U is indexed by [i, j] in 6x6, flattened
                # Value is sum over i_blk, j_blk of beta[s, i_blk, j_blk] * u_coeffs[(i_loc, j_loc)]
                for (i_loc, j_loc), val in u_coeffs.items():
                    for i_blk in range(2):
                        for j_blk in range(2):
                            coeff = beta[s, i_blk, j_blk]
                            if coeff != 0:
                                i = i_blk * 3 + i_loc
                                j = j_blk * 3 + j_loc
                                U[i * m + j, col_idx] += coeff * val

                # Fill V
                # V[row, col] where row = (j_blk * 3 + j_loc) * 6 + (k_blk * 3 + k_loc)
                # V is indexed by [j, k] in 6x6, flattened
                # Value is sum over j_blk, k_blk of gamma[s, j_blk, k_blk] * v_coeffs[(j_loc, k_loc)]
                for (j_loc, k_loc), val in v_coeffs.items():
                    for j_blk in range(2):
                        for k_blk in range(2):
                            coeff = gamma[s, j_blk, k_blk]
                            if coeff != 0:
                                j = j_blk * 3 + j_loc
                                k = k_blk * 3 + k_loc
                                V[j * p + k, col_idx] += coeff * val

                # Fill W
                # W[row, col] where row = (k_blk * 3 + k_loc) * 6 + (i_blk * 3 + i_loc)
                # W is indexed by [k, i] in 6x6, flattened
                # Value is sum over i_blk, k_blk of alpha[s, i_blk, k_blk] * w_coeffs[(i_loc, k_loc)]
                for (i_loc, k_loc), val in w_coeffs.items():
                    for i_blk in range(2):
                        for k_blk in range(2):
                            coeff = alpha[s, i_blk, k_blk]
                            if coeff != 0:
                                i = i_blk * 3 + i_loc
                                k = k_blk * 3 + k_loc
                                W[k * n + i, col_idx] += coeff * val

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Verify base patterns
    laderman_valid = verify_3x3_patterns(laderman_23_patterns)

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Kronecker M2 x M3 (rank 161)
    if time.time() < deadline - 30:
        kron_sol, kron_rank = build_kronecker_6x6()
        U_k = np.array(kron_sol["U"], dtype=float)
        V_k = np.array(kron_sol["V"], dtype=float)
        W_k = np.array(kron_sol["W"], dtype=float)
        err_k = verify_decomposition(U_k, V_k, W_k, A_test, B_test, C_expected)
        if err_k < 1e-10 and kron_rank < best_rank:
            best_solution = kron_sol
            best_rank = kron_rank

    # 2) Laderman-based 6x6 (rank 184)
    if (best_solution is None or best_rank > 184) and laderman_valid and time.time() < deadline - 30:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 3) Strassen-based 6x6 (rank 189)
    if (best_solution is None or best_rank > 189) and time.time() < deadline - 30:
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s, A_test, B_test, C_expected)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 4) Fallback: schoolbook (rank 216)
    if best_solution is None:
        schoolbook_sol, schoolbook_rank = build_schoolbook_6x6()
        best_solution = schoolbook_sol
        best_rank = schoolbook_rank

    return best_solution

# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
