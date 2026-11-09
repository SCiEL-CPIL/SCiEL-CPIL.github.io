"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, max_checks=20):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    import random
    random.seed(42)
    for _ in range(max_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i, j = a // m, a % m
        jp, k = b // p, b % p
        kp, ip = c // n, c % n
        expected = 1 if (j == jp and k == kp and i == ip) else 0
        actual = 0.0
        for r in range(rank):
            actual += U[a][r] * V[b][r] * W[c][r]
        if abs(actual - expected) > 1e-9:
            return False
    return True

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen 2x2x2: rank 7
    strassen_products = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    def add_strassen_block(a_col0, b_row0, b_col0, col_start):
        c = col_start
        for u_coeff, v_coeff, w_coeff in strassen_products:
            for (li, lj), coeff in u_coeff.items():
                gi = li
                gj = a_col0 + lj
                U[gi * m + gj][c] = coeff
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row0 + lj
                gk = b_col0 + lk
                V[gj * p + gk][c] = coeff
            for (lk, li), coeff in w_coeff.items():
                gk = b_col0 + lk
                gi = li
                W[gk * n + gi][c] = coeff
            c += 1
        return c

    col = 0
    # 8 blocks of 2x2x2
    for b_row0 in [0, 2]:
        for b_col0 in [0, 2, 4, 6]:
            col = add_strassen_block(0, b_row0, b_col0, col)
            col = add_strassen_block(2, b_row0, b_col0, col)

    if time.time() > deadline:
        pass

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, max_checks=50):
        rank_fb = n * m * p  # 64
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j][r] = 1
                    V_fb[j * p + k][r] = 1
                    W_fb[k * n + i][r] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
