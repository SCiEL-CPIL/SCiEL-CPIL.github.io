"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][col] = val
        for (i, k), val in w_coeffs.items():
            # W index is k*n + i
            W[k * n + i][col] = val
        col += 1

    # Strassen M222
    def strassen_222(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        # a_off_i, a_off_j: offsets for A in global i, j
        # b_off_j, b_off_k: offsets for B in global j, k
        # c_off_i, c_off_k: offsets for C in global i, k
        # Local indices 0,1
        
        # M1
        add_term({(a_off_i+0, a_off_j+0): 1, (a_off_i+1, a_off_j+1): 1},
                 {(b_off_j+0, b_off_k+0): 1, (b_off_j+1, b_off_k+1): 1},
                 {(c_off_i+0, c_off_k+0): 1, (c_off_i+1, c_off_k+1): 1})
        # M2
        add_term({(a_off_i+1, a_off_j+0): 1, (a_off_i+1, a_off_j+1): 1},
                 {(b_off_j+0, b_off_k+0): 1},
                 {(c_off_i+1, c_off_k+0): 1, (c_off_i+1, c_off_k+1): 1})
        # M3
        add_term({(a_off_i+0, a_off_j+0): 1},
                 {(b_off_j+0, b_off_k+1): 1, (b_off_j+1, b_off_k+1): -1},
                 {(c_off_i+0, c_off_k+1): 1})
        # M4
        add_term({(a_off_i+1, a_off_j+1): 1},
                 {(b_off_j+1, b_off_k+0): 1, (b_off_j+0, b_off_k+0): -1},
                 {(c_off_i+0, c_off_k+0): 1, (c_off_i+1, c_off_k+0): 1})
        # M5
        add_term({(a_off_i+0, a_off_j+0): 1, (a_off_i+0, a_off_j+1): 1},
                 {(b_off_j+1, b_off_k+1): 1},
                 {(c_off_i+0, c_off_k+0): -1, (c_off_i+0, c_off_k+1): 1})
        # M6
        add_term({(a_off_i+1, a_off_j+0): 1, (a_off_i+0, a_off_j+0): -1},
                 {(b_off_j+0, b_off_k+0): 1, (b_off_j+0, b_off_k+1): 1},
                 {(c_off_i+1, c_off_k+1): 1})
        # M7
        add_term({(a_off_i+0, a_off_j+1): 1, (a_off_i+1, a_off_j+1): -1},
                 {(b_off_j+1, b_off_k+0): 1, (b_off_j+1, b_off_k+1): 1},
                 {(c_off_i+0, c_off_k+0): 1, (c_off_i+1, c_off_k+1): 1})

    # M122 (1x2 x 2x2 -> 1x2)
    def m122(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        # A is 1x2, B is 2x2, C is 1x2
        # C_00 = A_00 B_00 + A_01 B_10
        # C_01 = A_00 B_01 + A_01 B_11
        
        add_term({(a_off_i+0, a_off_j+0): 1},
                 {(b_off_j+0, b_off_k+0): 1},
                 {(c_off_i+0, c_off_k+0): 1})
        add_term({(a_off_i+0, a_off_j+1): 1},
                 {(b_off_j+1, b_off_k+0): 1},
                 {(c_off_i+0, c_off_k+0): 1})
        add_term({(a_off_i+0, a_off_j+0): 1},
                 {(b_off_j+0, b_off_k+1): 1},
                 {(c_off_i+0, c_off_k+1): 1})
        add_term({(a_off_i+0, a_off_j+1): 1},
                 {(b_off_j+1, b_off_k+1): 1},
                 {(c_off_i+0, c_off_k+1): 1})

    # M321 (3x2 x 2x1 -> 3x1)
    def m321(a_off_i, a_off_j, b_off_j, b_off_k, c_off_i, c_off_k):
        # A is 3x2, B is 2x1, C is 3x1
        # C_i0 = A_i0 B_00 + A_i1 B_10
        for i in range(3):
            add_term({(a_off_i+i, a_off_j+0): 1},
                     {(b_off_j+0, b_off_k+0): 1},
                     {(c_off_i+i, c_off_k+0): 1})
            add_term({(a_off_i+i, a_off_j+1): 1},
                     {(b_off_j+1, b_off_k+0): 1},
                     {(c_off_i+i, c_off_k+0): 1})

    # Block 1: A1 (cols 0-1) B1a (rows 0-1, cols 0-1) -> C (cols 0-1)
    # M322 = M222 (top) + M122 (bot)
    # A1 top: i 0-1, j 0-1. B1a: j 0-1, k 0-1. C top: i 0-1, k 0-1.
    strassen_222(0, 0, 0, 0, 0, 0)
    # A1 bot: i 2, j 0-1. B1a: j 0-1, k 0-1. C bot: i 2, k 0-1.
    m122(2, 0, 0, 0, 2, 0)

    # Block 2: A1 (cols 0-1) B1b (rows 0-1, cols 2-3) -> C (cols 2-3)
    strassen_222(0, 0, 0, 2, 0, 2)
    m122(2, 0, 0, 2, 2, 2)

    # Block 3: A1 (cols 0-1) B1c (rows 0-1, cols 4) -> C (cols 4)
    m321(0, 0, 0, 4, 0, 4)

    # Block 4: A2 (cols 2-3) B2a (rows 2-3, cols 0-1) -> C (cols 0-1)
    strassen_222(0, 2, 2, 0, 0, 0)
    m122(2, 2, 2, 0, 2, 0)

    # Block 5: A2 (cols 2-3) B2b (rows 2-3, cols 2-3) -> C (cols 2-3)
    strassen_222(0, 2, 2, 2, 0, 2)
    m122(2, 2, 2, 2, 2, 2)

    # Block 6: A2 (cols 2-3) B2c (rows 2-3, cols 4) -> C (cols 4)
    m321(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
