"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    1-level recursive Strassen for 6x6 matrix multiplication.
    
    Strategy:
    1. Split 6x6 matrices into 2x2 blocks of 3x3 submatrices.
    2. Apply Strassen's algorithm to the 2x2 block structure (7 multiplications).
    3. Each of the 7 block multiplications is a 3x3 matrix multiplication.
    4. Use schoolbook algorithm for the 3x3 multiplications (27 scalar mults each).
    
    Total Rank = 7 * 27 = 189.
    This is better than schoolbook 6x6 (Rank 216).
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # 7 Strassen combinations at level 1 for 6x6 with 3x3 blocks
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),   # T1
        ({(1,0):1, (1,1):1}, {(0,0):1}),             # T2
        ({(0,0):1}, {(0,1):1, (1,1):-1}),            # T3
        ({(1,1):1}, {(1,0):1, (0,0):-1}),            # T4
        ({(0,0):1, (0,1):1}, {(1,1):1}),             # T5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),   # T6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),   # T7
    ]
    
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
