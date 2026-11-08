import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # Tiled 2x2x2 for 6x6: total rank = 8 * 23 = 184
    block_size = 3
    blocks_per_dim = 2  # 6 / 3

    rank = (blocks_per_dim ** 3) * 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Laderman's 23 rank-1 terms for 3x3 matrix multiplication
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: {(i,j): val} for A (3x3)
    # v_coeffs: {(j,k): val} for B (3x3)
    # w_coeffs: {(i,k): val} for C (3x3)
    #
    # Verified Laderman decomposition:
    laderman_patterns = [
        # 1: (A00+A22)(B00+B22) -> C00 +=, C22 +=, C02 -=, C20 -=
        ({(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1},
         {(0,0): 1, (2,2): 1, (0,2): -1, (2,0): -1}),
        # 2: (A00+A11)(B00+B11) -> C00 +=, C11 +=, C01 -=, C10 -=
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1}),
        # 3: (A00+A22)(B01+B21) -> C01 +=, C21 +=, C00 -=, C20 -=  ... wait let me use standard
        # Let me use the standard Laderman from literature:
    ]

    # Standard Laderman's algorithm (23 multiplications for 3x3):
    # Using the well-known formulation:
    laderman_patterns = [
        # 1
        ({(0,0): 1, (2,2): 1}, {(0,0): 1, (2,2): 1},
         {(0,0): 1, (2,2): 1, (0,2): -1, (2,0): -1}),
        # 2
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1}),
        # 3
        ({(0,0): 1, (2,2): 1}, {(0,1): 1, (2,1): 1},
         {(0,1): 1, (2,1): 1, (0,0): -1, (2,0): -1}),
        # 4
        ({(0,0): 1, (1,1): 1}, {(0,1): 1, (1,1): 1},
         {(0,1): 1, (1,1): 1, (0,0): -1, (1,0): -1}),
        # 5
        ({(0,0): 1, (2,2): 1}, {(1,0): 1, (2,0): 1},
         {(1,0): 1, (2,0): 1, (0,0): -1, (2,0): -1}),  # hmm this seems off
    ]

    # I need to be very careful. Let me use a different approach.
    # The safest bet is to verify the Strassen 189 solution works,
    # and also try the Laderman tiling.
    #
    # Actually, let me just go with the proven 189 Strassen solution
    # since I cannot verify Laderman coefficients with 100% confidence
    # in this context. The 189 solution is known to be correct.

    # FALLBACK: Use the verified Strassen 2x2 block tiling (rank 189)
    block_size = 2
    blocks_per_dim = 3  # 6 / 2

    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_patterns = [
        # M0 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M1 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M2 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M3 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M4 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M5 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A01-A11)(B10+B11)
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
                        U[r * m + c][col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
