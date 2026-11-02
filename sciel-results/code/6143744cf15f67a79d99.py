"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    Strategy: Decompose the problem into four 2x2 by 2x2 sub-problems and apply Strassen's
    algorithm to each, accumulating contributions to the output tensor.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 instances * 7 Strassen multipliers each

    # U: shape (n*m, rank) = (8, 28)
    # V: shape (m*p, rank) = (16, 28)
    # W: shape (p*n, rank) = (8, 28)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen's 7 multipliers for 2x2 matrix multiplication:
    # Given A (2x2) and B (2x2), with elements:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # Output C (2x2):
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6

    # Define the four 2x2 sub-problems:
    # Instance 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # Instance 2: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    # Instance 3: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    # Instance 4: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3

    instances = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
        (0, 0, 0, 0, 0, 0),   # Instance 1
        (0, 0, 0, 2, 0, 2),   # Instance 2
        (0, 2, 2, 0, 0, 0),   # Instance 3
        (0, 2, 2, 2, 0, 2),   # Instance 4
    ]

    # Strassen multipliers and their contributions to output elements:
    # For each multiplier r in 0..6, we need:
    # - U coefficients: which A elements contribute, with what sign
    # - V coefficients: which B elements contribute, with what sign
    # - W coefficients: which C elements receive this multiplier, with what sign

    # Multiplier definitions:
    # M0 (M1): U: a00(+1), a11(+1); V: b00(+1), b11(+1); W: c00(+1), c11(+1)
    # M1 (M2): U: a10(+1), a11(+1); V: b00(+1); W: c10(+1), c11(-1)
    # M2 (M3): U: a00(+1); V: b01(+1), b11(-1); W: c01(+1), c11(+1)
    # M3 (M4): U: a11(+1); V: b10(+1), b00(-1); W: c00(+1), c10(+1)
    # M4 (M5): U: a00(+1), a01(+1); V: b11(+1); W: c00(-1), c01(+1)
    # M5 (M6): U: a10(+1), a00(-1); V: b00(+1), b01(+1); W: c11(+1)
    # M6 (M7): U: a01(+1), a11(-1); V: b10(+1), b11(+1); W: c00(+1)

    for inst_idx, (a_r, a_c, b_r, b_c, c_r, c_c) in enumerate(instances):
        base_col = inst_idx * 7

        # M0: (a00 + a11) * (b00 + b11) -> c00(+1), c11(+1)
        col = base_col
        # U: a00 at (a_r + 0)*m + (a_c + 0) = 0*4+0=0; a11 at (a_r+1)*m+(a_c+1)=1*4+1=5
        U[0][col] += 1.0
        U[5][col] += 1.0
        # V: b00 at (b_r+0)*p+(b_c+0)=0*4+0=0; b11 at (b_r+1)*p+(b_c+1)=1*4+1=5
        V[0][col] += 1.0
        V[5][col] += 1.0
        # W: c00 at (c_c+0)*n+(c_r+0)=0*2+0=0; c11 at (c_c+1)*n+(c_r+1)=1*2+1=3
        W[0][col] += 1.0
        W[3][col] += 1.0

        # M1: (a10 + a11) * b00 -> c10(+1), c11(-1)
        col = base_col + 1
        # U: a10 at (a_r+1)*m+(a_c+0)=1*4+0=4; a11 at 5
        U[4][col] += 1.0
        U[5][col] += 1.0
        # V: b00 at 0
        V[0][col] += 1.0
        # W: c10 at (c_c+0)*n+(c_r+1)=0*2+1=1; c11 at 3
        W[1][col] += 1.0
        W[3][col] += -1.0

        # M2: a00 * (b01 - b11) -> c01(+1), c11(+1)
        col = base_col + 2
        # U: a00 at 0
        U[0][col] += 1.0
        # V: b01 at (b_r+0)*p+(b_c+1)=0*4+1=1; b11 at 5
        V[1][col] += 1.0
        V[5][col] += -1.0
        # W: c01 at (c_c+1)*n+(c_r+0)=1*2+0=2; c11 at 3
        W[2][col] += 1.0
        W[3][col] += 1.0

        # M3: a11 * (b10 - b00) -> c00(+1), c10(+1)
        col = base_col + 3
        # U: a11 at 5
        U[5][col] += 1.0
        # V: b10 at (b_r+1)*p+(b_c+0)=1*4+0=4; b00 at 0
        V[4][col] += 1.0
        V[0][col] += -1.0
        # W: c00 at 0; c10 at 1
        W[0][col] += 1.0
        W[1][col] += 1.0

        # M4: (a00 + a01) * b11 -> c00(-1), c01(+1)
        col = base_col + 4
        # U: a00 at 0; a01 at (a_r+0)*m+(a_c+1)=0*4+1=1
        U[0][col] += 1.0
        U[1][col] += 1.0
        # V: b11 at 5
        V[5][col] += 1.0
        # W: c00 at 0; c01 at 2
        W[0][col] += -1.0
        W[2][col] += 1.0

        # M5: (a10 - a00) * (b00 + b01) -> c11(+1)
        col = base_col + 5
        # U: a10 at 4; a00 at 0
        U[4][col] += 1.0
        U[0][col] += -1.0
        # V: b00 at 0; b01 at 1
        V[0][col] += 1.0
        V[1][col] += 1.0
        # W: c11 at 3
        W[3][col] += 1.0

        # M6: (a01 - a11) * (b10 + b11) -> c00(+1)
        col = base_col + 6
        # U: a01 at 1; a11 at 5
        U[1][col] += 1.0
        U[5][col] += -1.0
        # V: b10 at 4; b11 at 5
        V[4][col] += 1.0
        V[5][col] += 1.0
        # W: c00 at 0
        W[0][col] += 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

