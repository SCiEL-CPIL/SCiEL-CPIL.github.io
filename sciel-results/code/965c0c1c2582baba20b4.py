import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-56 decomposition using Strassen's algorithm on 2x2 blocks.
    # The multiplication C = A B (2x4 * 4x8) is decomposed into 4 independent
    # 2x4 * 4x2 multiplications (slicing B into 4 blocks of width 2).
    # Each 2x4 * 4x2 multiplication is split into two 2x2 * 2x2 multiplications
    # (slicing A and B into halves along the m=4 dimension).
    # Each 2x2 * 2x2 uses Strassen (rank 7).
    # Total rank = 4 blocks * 2 halves * 7 = 56.

    # Strassen 2x2 local factorization (rank 7).
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    group = 0
    for jj in range(p // 2):
        for half in range(2):
            base_r = group * 7
            group += 1

            # Map local indices to global indices
            # A: row i, col j -> i*m + j
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            # B: row i, col j -> i*p + j
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            # W: col k, row i -> k*n + i
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
