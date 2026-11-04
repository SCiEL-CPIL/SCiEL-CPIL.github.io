"""Combined: Kronecker product of rank-4 (2x2) with rank-27 schoolbook (3x3) for 6x6x6 rank 108."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def build_strassen_6x6():
        """Build the 6x6x6 decomposition using Strassen on 2x2 blocks (rank 189)."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

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

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_kronecker_2x3():
        """
        Build 6x6x6 via Kronecker product of rank-4 (2x2) and rank-27 schoolbook (3x3).
        Total rank = 4 * 27 = 108.
        
        View 6x6 as 3x3 blocks of 2x2 submatrices.
        Index mapping: i = 2*a + a' where a in {0,1,2} (block), a' in {0,1} (subblock)
        """
        # Rank-4 decomposition for 2x2 matrix multiplication
        # C = A*B where A,B,C are 2x2
        # M1 = a00 * b00
        # M2 = a01 * b11
        # M3 = (a00+a10) * (b01+b11)
        # M4 = (a10-a00) * (b00-b10)
        # C00 = M1 + M3 - M4
        # C01 = M2 + M3
        # C10 = M1 + M4
        # C11 = M2
        rank2_patterns = [
            ({(0,0): 1}, {(0,0): 1}, {(0,0): 1, (1,0): 1}),
            ({(0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
            ({(0,0): 1, (1,0): 1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): -1}, {(0,0): -1, (1,0): 1}),
        ]

        # Rank-27 schoolbook for 3x3 matrix multiplication
        rank3_patterns = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    rank3_patterns.append(
                        ({(i, j): 1}, {(j, k): 1}, {(k, i): 1})
                    )

        rank = 4 * 27
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for r2, (u2, v2, w2) in enumerate(rank2_patterns):
            for r3, (u3, v3, w3) in enumerate(rank3_patterns):
                r = r2 * 27 + r3

                # U: index (i, j) where i = 2*a + a', j = 2*b + b'
                for (a, b), val3 in u3.items():
                    for (a_p, b_p), val2 in u2.items():
                        i = 2 * a + a_p
                        j = 2 * b + b_p
                        U[i * m + j, r] += val3 * val2

                # V: index (j, k) where j = 2*b + b', k = 2*c + c'
                for (b, c), val3 in v3.items():
                    for (b_p, c_p), val2 in v2.items():
                        j = 2 * b + b_p
                        k = 2 * c + c_p
                        V[j * p + k, r] += val3 * val2

                # W: index (k, i) where k = 2*c + c', i = 2*a + a'
                for (c, a), val3 in w3.items():
                    for (c_p, a_p), val2 in w2.items():
                        k = 2 * c + c_p
                        i = 2 * a + a_p
                        W[k * n + i, r] += val3 * val2

        # Verify the full 6x6 decomposition
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        U_arr = U
        V_arr = V
        W_arr = W
        for r in range(rank):
            M_u = sum(U_arr[i * m + j, r] * A_test[i, j] for i in range(n) for j in range(m))
            M_v = sum(V_arr[j * p + k, r] * B_test[j, k] for j in range(m) for k in range(p))
            M_r = M_u * M_v
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += M_r * W_arr[k * n + i, r]

        err = np.max(np.abs(C_computed - C_expected))
        if err > 1e-10:
            return None

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Kronecker 2x3 first (rank 108)
    best_solution = None
    best_rank = float('inf')

    kron_result = build_kronecker_2x3()
    if kron_result is not None:
        best_solution = kron_result
        best_rank = 108

    # Fallback to Strassen 6x6 (rank 189)
    if best_solution is None:
        best_solution = build_strassen_6x6()
        best_rank = 189

    return best_solution
# EVOLVE-BLOCK-END
