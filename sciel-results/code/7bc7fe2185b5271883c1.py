"""Matrix multiplication 2x3x10: block-Kronecker rank-61 baseline."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 base tensor: rows U = i*2+j, cols V = j*2+k, rows W = k*2+i
# Each term (r,c) has U_mask, V_mask, W_mask (each a list of 3 (row,col) pairs, or empty).
_STRASSEN = [
    ([(0, 0), (0, 1), (1, 0), (1, 1)],
     [(0, 0), (1, 0)],
     [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ([(0, 0), (0, 1), (1, 1)],
     [(0, 1), (1, 1)],
     [(0, 1), (1, 1)]),
    ([(0, 0)],
     [(0, 0)],
     [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ([(1, 1)],
     [(0, 1)],
     [(0, 1), (1, 0), (1, 1)]),
    ([(0, 0), (0, 1), (1, 1)],
     [(0, 0)],
     [(0, 0), (0, 1)]),
    ([(1, 0), (1, 1)],
     [(0, 1), (1, 1)],
     [(1, 0), (1, 1)]),
    ([(0, 0), (0, 1), (1, 0)],
     [(1, 0)],
     [(1, 0), (1, 1)]),
]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 7 + 5 * 10  # 61: Strassen on (2x2) block + 5 rank-10 schoolbook terms

    U = [[0] * rank for _ in range(n * m)]  # 6 x 61
    V = [[0] * rank for _ in range(m * p)]  # 30 x 61
    W = [[0] * rank for _ in range(p * n)]  # 20 x 61

    col = 0
    # --- Strassen block: rows i in {0,1}, middle j in {0,1}, cols k in {0,1} ---
    for (u_rows, v_rows, w_rows) in _STRASSEN:
        for (ii, jj) in u_rows:
            U[ii * m + jj][col] = 1
        for (jj, kk) in v_rows:
            V[jj * p + kk][col] = 1
        for (kk, ii) in w_rows:
            W[kk * n + ii][col] = 1
        col += 1

    # --- Remaining 8 columns (k = 2..9): 5 rank-10 schoolbook terms ---
    for t in range(5):
        i = t % 2
        j = t // 2
        for k in range(2, p):
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
