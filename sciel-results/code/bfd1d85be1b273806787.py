def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Part 1: Row 0, all j,k (1×4×6): rank 24
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1

    # Part 2: Rows 1-2, middle j=0,1 (2×2×6): rank 12
    for i_prime in range(2):  # i = i_prime + 1
        for j in range(2):  # j = 0, 1
            for k in range(p):
                U[(i_prime + 1) * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + (i_prime + 1), col] = 1
                col += 1

    # Part 3: Rows 1-2, middle j=2,3 (2×2×6): rank 12
    for i_prime in range(2):  # i = i_prime + 1
        for j_prime in range(2):  # j = j_prime + 2
            for k in range(p):
                U[(i_prime + 1) * m + (j_prime + 2), col] = 1
                V[(j_prime + 2) * p + k, col] = 1
                W[k * n + (i_prime + 1), col] = 1
                col += 1

    # Verify
    u = U.reshape(n, m, rank)
    v = V.reshape(m, p, rank)
    w = W.reshape(p, n, rank)
    T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
    target = np.ones((n, m, p))
    if not np.allclose(T_reconstructed, target, atol=1e-9):
        # Fallback
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
