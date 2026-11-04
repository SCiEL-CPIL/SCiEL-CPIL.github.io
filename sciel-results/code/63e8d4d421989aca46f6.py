"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # We decompose the 4x4x6 matrix multiplication into 12 independent 2x2x2
    # sub-problems using Strassen's rank-7 algorithm for each.
    #
    # Block structure:
    # A (4x4) is divided into 2x2 blocks of size 2x2.
    #   A[I, J] where I, J in {0, 1}
    # B (4x6) is divided into 2x3 blocks of size 2x2.
    #   B[J, K] where J in {0, 1}, K in {0, 1, 2}
    # C (4x6) is divided into 2x3 blocks of size 2x2.
    #   C[I, K] where I in {0, 1}, K in {0, 1, 2}
    #
    # C[I, K] = sum_{J=0}^{1} A[I, J] * B[J, K]
    #
    # For each (I, K) pair, we have a 2x2 matrix multiplication over the ring
    # of 2x2 matrices. This can be decomposed as:
    #   C[I, K] = A[I, 0] * B[0, K] + A[I, 1] * B[1, K]
    #
    # Each product A[I, J] * B[J, K] is a 2x2x2 multiplication.
    # We use Strassen's rank-7 decomposition for each.
    #
    # Total number of 2x2x2 multiplications: 2 * 3 * 2 = 12
    # Total rank: 12 * 7 = 84

    rank = 84

    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84

    # Strassen coefficients for 2x2x2 multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b01-b11)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U coefficients (for A) - local indices: 00, 01, 10, 11
    U_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [1, 0, -1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]

    # V coefficients (for B) - local indices: 00, 01, 10, 11
    V_coeffs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [0, 1, 0, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]

    # W coefficients (for C) - local indices in trace convention: k*n+i
    # For a 2x2 block, local indices are:
    #   C00 -> index 0 (k=0, i=0)
    #   C10 -> index 1 (k=0, i=1)
    #   C01 -> index 2 (k=1, i=0)
    #   C11 -> index 3 (k=1, i=1)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    W_coeffs = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 1, 0, -1],  # M2: C10 += 1, C11 -= 1
        [0, 0, 1, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 0, 0],   # M4: C00 += 1
        [-1, 0, 1, 0],  # M5: C00 -= 1, C01 += 1
        [0, 1, 0, 1],   # M6: C10 += 1, C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]

    col = 0
    for I in range(2):
        for K in range(3):
            for J in range(2):
                for t in range(7):
                    # U: A[I, J] block
                    # Global row index: 2*I + local_row
                    # Global col index: 2*J + local_col
                    # U index: row * m + col = (2*I + local_row) * 4 + (2*J + local_col)
                    for u_local in range(4):
                        local_row = u_local // 2
                        local_col = u_local % 2
                        global_row = 2 * I + local_row
                        global_col = 2 * J + local_col
                        u_idx = global_row * m + global_col
                        U[u_idx][col] += U_coeffs[t][u_local]

                    # V: B[J, K] block
                    # Global row index: 2*J + local_row
                    # Global col index: 2*K + local_col
                    # V index: row * p + col = (2*J + local_row) * 6 + (2*K + local_col)
                    for v_local in range(4):
                        local_row = v_local // 2
                        local_col = v_local % 2
                        global_row = 2 * J + local_row
                        global_col = 2 * K + local_col
                        v_idx = global_row * p + global_col
                        V[v_idx][col] += V_coeffs[t][v_local]

                    # W: C[I, K] block
                    # Global row index: 2*I + local_row
                    # Global col index: 2*K + local_col
                    # W index: k*n + i = global_col * n + global_row = (2*K + local_col) * 4 + (2*I + local_row)
                    for w_local in range(4):
                        local_row = w_local // 2
                        local_col = w_local % 2
                        global_row = 2 * I + local_row
                        global_col = 2 * K + local_col
                        w_idx = global_col * n + global_row
                        W[w_idx][col] += W_coeffs[t][w_local]

                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
