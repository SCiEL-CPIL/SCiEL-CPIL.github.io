"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0

    n, m, p = DIMENSIONS  # 2, 5, 6

    # Strassen 2x2 x 2x2 factors:
    # (U_coeffs, V_coeffs, W_coeffs)
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]

    rank = 54
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    col = 0

    def add_strassen_block(a_rows, b_rows, c_cols, col):
        for u_c, v_c, w_c in strassen_factors:
            for (ai, aj), val in u_c.items():
                U[a_rows[ai]*m + b_rows[aj]][col] += val
            for (bj, bk), val in v_c.items():
                V[b_rows[bj]*p + c_cols[bk]][col] += val
            for (ck, ci), val in w_c.items():
                W[c_cols[ck]*n + a_rows[ci]][col] += val
            col += 1
        return col

    # 2x2 x 2x6 blocks (j=0,1 and j=2,3), each split into 3 Strassen 2x2x2x2
    for b_pair in ([0,1], [2,3]):
        for c_pair in ([0,1], [2,3], [4,5]):
            col = add_strassen_block([0,1], b_pair, c_pair, col)

    # 2x1 x 1x6 block (j=4), schoolbook
    for i in range(n):
        for k in range(p):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1

    assert col == rank

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
