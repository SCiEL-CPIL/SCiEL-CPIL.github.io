import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit

    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen factors for 2x2 matrix multiplication
    # A flat: [A[0,0], A[0,1], A[1,0], A[1,1]]
    # B flat: [B[0,0], B[0,1], B[1,0], B[1,1]]
    # C flat: [C[0,0], C[1,0], C[0,1], C[1,1]]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    col = 0
    for blk in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A indices [i*m + k] = [i*2 + k]
            for idx in range(4):
                U[idx][col] = a_vec[idx]

            # V: B indices [k*p + j], block covers cols 2*blk, 2*blk+1
            V[0 * p + 2 * blk][col] = b_vec[0]
            V[0 * p + 2 * blk + 1][col] = b_vec[1]
            V[1 * p + 2 * blk][col] = b_vec[2]
            V[1 * p + 2 * blk + 1][col] = b_vec[3]

            # W: C indices [j*n + i], j is col of C, i is row of C
            W[2 * blk * n + 0][col] = c_vec[0]
            W[2 * blk * n + 1][col] = c_vec[1]
            W[(2 * blk + 1) * n + 0][col] = c_vec[2]
            W[(2 * blk + 1) * n + 1][col] = c_vec[3]

            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
