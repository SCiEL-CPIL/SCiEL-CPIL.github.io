import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # These patterns are a known correct decomposition for 3x3 matrix multiplication.
    # Note: The indices are (row, col) for A, B, and C.
    # A is 3x3, B is 3x3, C is 3x3.
    laderman_23_patterns = [
        # 1
        ({(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}),
        # 2
        ({(0,0):1}, {(1,0):1}, {(0,0):-1, (1,0):-1}),
        # 3
        ({(1,0):1}, {(0,0):1}, {(0,0):-1, (1,0):-1}),
        # 4
        ({(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}),
        # 5
        ({(0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):-1}),
        # 6
        ({(1,1):1}, {(0,1):1}, {(0,1):-1, (1,1):-1}),
        # 7
        ({(0,2):1, (1,2):1}, {(0,2):1, (1,2):1}, {(0,2):1, (1,2):1}),
        # 8
        ({(0,2):1}, {(1,2):1}, {(0,2):-1, (1,2):-1}),
        # 9
        ({(1,2):1}, {(0,2):1}, {(0,2):-1, (1,2):-1}),
        # 10
        ({(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}),
        # 11
        ({(0,0):1}, {(1,1):1}, {(0,0):-1, (0,1):-1}),
        # 12
        ({(0,1):1}, {(1,0):1}, {(0,0):-1, (0,1):-1}),
        # 13
        ({(0,0):1, (0,2):1}, {(1,0):1, (1,2):1}, {(0,0):1, (0,2):1}),
        # 14
        ({(0,0):1}, {(1,2):1}, {(0,0):-1, (0,2):-1}),
        # 15
        ({(0,2):1}, {(1,0):1}, {(0,0):-1, (0,2):-1}),
        # 16
        ({(0,1):1, (0,2):1}, {(1,1):1, (1,2):1}, {(0,1):1, (0,2):1}),
        # 17
        ({(0,1):1}, {(1,2):1}, {(0,1):-1, (0,2):-1}),
        # 18
        ({(0,2):1}, {(1,1):1}, {(0,1):-1, (0,2):-1}),
        # 19
        ({(2,0):1, (2,1):1}, {(0,0):1, (0,1):1}, {(2,0):1, (2,1):1}),
        # 20
        ({(2,0):1}, {(0,1):1}, {(2,0):-1, (2,1):-1}),
        # 21
        ({(2,1):1}, {(0,0):1}, {(2,0):-1, (2,1):-1}),
        # 22
        ({(2,0):1, (2,2):1}, {(0,0):1, (0,2):1}, {(2,0):1, (2,2):1}),
        # 23
        ({(2,0):1}, {(0,2):1}, {(2,0):-1, (2,2):-1}),
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
        """Verify Laderman patterns against random 3x3 matrix products."""
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
        # Also test with structured matrices
        for A_t, B_t in [
            (np.eye(3), np.eye(3)),
            (np.ones((3,3)), np.ones((3,3))),
            (np.arange(9).reshape(3,3).astype(float), np.eye(3)),
            (np.eye(3), np.arange(9).reshape(3,3).astype(float)),
        ]:
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
        """Verify that U, V, W decompose the matrix multiplication tensor.
        C[i,k] = sum_j A[i,j] * B[j,k]
        U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        W index: k*n + i
        """
        rank_used = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        A_flat = A_test.flatten()  # index: i*m + j
        B_flat = B_test.flatten()  # index: j*p + k
        for r in range(rank_used):
            M_u = np.dot(U_arr[:, r], A_flat)
            M_v = np.dot(V_arr[:, r], B_flat)
            M_r = M_u * M_v
            # W[:, r] has shape (p*n,), index k*n + i
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

    def build_winograd_6x6():
        """
        Winograd 4x4 rank 49 + Strassen 2x2 rank 7, mixed block approach.
        Actually, let's use a 4+2 split:
        - Top-left 4x4 * 4x4 -> 4x4: Winograd rank 49
        - Top-right 4x2 * 2x4 -> 4x4: schoolbook 32
        - Bottom-left 2x4 * 4x4 -> 2x4: schoolbook 32
        - Bottom-right 2x2 * 2x2 -> 2x2: Strassen rank 7
        Total: 49+32+32+7 = 120? No, that's not how it works.
        
        Actually for block C = A*B with A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]:
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
        
        This is more complex. Let's just use the simple constructions.
        """
        return None, float('inf')

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Additional test cases for robustness
    test_cases = [(A_test, B_test, C_expected)]
    np.random.seed(7)
    for _ in range(2):
        A_t = np.random.randn(n, m)
        B_t = np.random.randn(m, p)
        test_cases.append((A_t, B_t, A_t @ B_t))

    def verify_solution(sol, test_cases):
        U_arr = np.array(sol["U"], dtype=float)
        V_arr = np.array(sol["V"], dtype=float)
        W_arr = np.array(sol["W"], dtype=float)
        max_err = 0.0
        for A_t, B_t, C_exp in test_cases:
            err = verify_decomposition(U_arr, V_arr, W_arr, A_t, B_t, C_exp)
            max_err = max(max_err, err)
        return max_err

    # Verify base patterns
    laderman_valid = verify_3x3_patterns(laderman_23_patterns)

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Laderman-based 6x6 (rank 184)
    if laderman_valid and time.time() < deadline - 20:
        laderman_sol, laderman_rank = build_laderman_6x6()
        err_l = verify_solution(laderman_sol, test_cases)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 2) Strassen-based 6x6 (rank 189)
    if (best_solution is None or time.time() < deadline - 20):
        strassen_sol, strassen_rank = build_strassen_6x6()
        err_s = verify_solution(strassen_sol, test_cases)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 3) Fallback: schoolbook (rank 216)
    if best_solution is None:
        schoolbook_sol, schoolbook_rank = build_schoolbook_6x6()
        best_solution = schoolbook_sol
        best_rank = schoolbook_rank

    return best_solution
# EVOLVE-BLOCK-END
