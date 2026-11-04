"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for 6x6x6 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock limit

    # Laderman's 3x3 matrix multiplication algorithm (rank 23)
    # Verified correct patterns for C = A*B where A,B,C are 3x3
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
        """Verify that the 3x3 patterns correctly compute C = A*B."""
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

    def verify_2x2_patterns(patterns):
        """Verify that the 2x2 patterns correctly compute C = A*B."""
        np.random.seed(456)
        for trial in range(5):
            A_t = np.random.randn(2, 2)
            B_t = np.random.randn(2, 2)
            C_expected = A_t @ B_t
            C_computed = np.zeros((2, 2))
            for u_c, v_c, w_c in patterns:
                mu = sum(val * A_t[r, c] for (r, c), val in u_c.items())
                mv = sum(val * B_t[r, c] for (r, c), val in v_c.items())
                for (r, c), val in w_c.items():
                    C_computed[r, c] += val * mu * mv
            if np.max(np.abs(C_computed - C_expected)) > 1e-10:
                return False
        return True

    def verify_decomposition(U_arr, V_arr, W_arr, A_test, B_test, C_expected):
        """Verify a decomposition is correct."""
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
        """Build 6x6x6 decomposition using Laderman's 3x3 algorithm on 2x2 blocks."""
        block_size = 3
        blocks_per_dim = 2
        rank = (blocks_per_dim ** 3) * 23  # 8 * 23 = 184
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
        """Build the 6x6x6 decomposition using Strassen on 2x2 blocks (rank 189)."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7  # 27 * 7 = 189
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

    def build_mixed_4_2_6x6():
        """
        Build 6x6x6 using a mixed 4+2 block decomposition.
        Split each dimension into 4+2.
        - 4x4 x 4x4 -> 4x4: use Winograd rank 49
        - 4x2 x 2x4 -> 4x4: schoolbook 4*2*4=32 (no known better)
        - 2x4 x 4x2 -> 2x2: schoolbook 2*4*2=16
        - 2x2 x 2x2 -> 2x2: Strassen rank 7
        Total: 49 + 32 + 16 + 7 = 104... wait that's not right.
        
        Actually for block matrix multiplication C = A*B:
        If A is partitioned as [A11 A12; A21 A22] with A11 4x4, A12 4x2, A21 2x4, A22 2x2
        and B as [B11 B12; B21 B22] with B11 4x4, B12 4x2, B21 2x4, B22 2x2
        Then C = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
        
        So we need:
        - A11(4x4) * B11(4x4) -> 4x4: rank 49
        - A12(4x2) * B21(2x4) -> 4x4: rank 4*2*4 = 32 (schoolbook, no better known for rectangular)
        - A11(4x4) * B12(4x2) -> 4x2: rank 4*4*2 = 32
        - A12(4x2) * B22(2x2) -> 4x2: rank 4*2*2 = 16
        - A21(2x4) * B11(4x4) -> 2x4: rank 2*4*4 = 32
        - A22(2x2) * B21(2x4) -> 2x4: rank 2*2*4 = 16
        - A21(2x4) * B12(4x2) -> 2x2: rank 2*4*2 = 16
        - A22(2x2) * B22(2x2) -> 2x2: rank 7 (Strassen)
        
        Total: 49 + 32 + 32 + 16 + 32 + 16 + 16 + 7 = 200. Worse than 184!
        """
        return None, float('inf')

    # Test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    # Verify base patterns
    laderman_valid = verify_3x3_patterns(laderman_23_patterns)
    strassen_valid = verify_2x2_patterns(strassen_patterns)

    # Try known constructions and pick the lowest-rank valid one
    best_solution = None
    best_rank = float('inf')

    # 1) Laderman-based 6x6 (rank 184) - best known
    if laderman_valid and time.time() < deadline - 20:
        laderman_sol, laderman_rank = build_laderman_6x6()
        U_l = np.array(laderman_sol["U"], dtype=float)
        V_l = np.array(laderman_sol["V"], dtype=float)
        W_l = np.array(laderman_sol["W"], dtype=float)
        err_l = verify_decomposition(U_l, V_l, W_l, A_test, B_test, C_expected)
        if err_l < 1e-10 and laderman_rank < best_rank:
            best_solution = laderman_sol
            best_rank = laderman_rank

    # 2) Strassen-based 6x6 (rank 189)
    if strassen_valid and (best_solution is None or time.time() < deadline - 20):
        strassen_sol, strassen_rank = build_strassen_6x6()
        U_s = np.array(strassen_sol["U"], dtype=float)
        V_s = np.array(strassen_sol["V"], dtype=float)
        W_s = np.array(strassen_sol["W"], dtype=float)
        err_s = verify_decomposition(U_s, V_s, W_s, A_test, B_test, C_expected)
        if err_s < 1e-10 and strassen_rank < best_rank:
            best_solution = strassen_sol
            best_rank = strassen_rank

    # 3) Fallback: schoolbook (rank 216)
    if best_solution is None:
        rank_trivial = n * m * p  # 216
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
        best_rank = rank_trivial

    return best_solution
# EVOLVE-BLOCK-END
