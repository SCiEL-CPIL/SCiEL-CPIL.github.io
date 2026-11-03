"""Matrix multiplication 2x3x4 with Strassen-based rank reduction (rank 22)."""
DIMENSIONS = [2, 3, 4]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22  # 2 blocks x (7 Strassen + 4 extra) = 22

    U = [[0.0] * rank for _ in range(n * m)]  # (6, 22)
    V = [[0.0] * rank for _ in range(m * p)]  # (12, 22)
    W = [[0.0] * rank for _ in range(p * n)]  # (8, 22)

    # Strassen terms for 2x2 matrix multiplication (i,j,k all in {0,1}).
    # Each term: (u_part, v_part, w_part)
    #   u_part: dict (i, j) -> coeff   [indices into A: row i, col j]
    #   v_part: dict (j, k) -> coeff   [indices into B: row j, col k]
    #   w_part: dict (k, i) -> coeff   [indices into C: col k, row i]
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
        (
            {(0, 0): 1.0, (1, 1): 1.0},
            {(0, 0): 1.0, (1, 1): 1.0},
            {(0, 0): 1.0, (1, 1): 1.0},
        ),
        # M2 = (A10+A11)B00: C10 += M2, C11 -= M2
        (
            {(1, 0): 1.0, (1, 1): 1.0},
            {(0, 0): 1.0},
            {(0, 1): 1.0, (1, 1): -1.0},
        ),
        # M3 = A00(B01-B11): C01 += M3, C11 += M3
        (
            {(0, 0): 1.0},
            {(0, 1): 1.0, (1, 1): -1.0},
            {(1, 0): 1.0, (1, 1): 1.0},
        ),
        # M4 = A11(B10-B00): C00 += M4, C10 += M4
        (
            {(1, 1): 1.0},
            {(1, 0): 1.0, (0, 0): -1.0},
            {(0, 0): 1.0, (0, 1): 1.0},
        ),
        # M5 = (A00+A01)B11: C00 -= M5, C01 += M5
        (
            {(0, 0): 1.0, (0, 1): 1.0},
            {(1, 1): 1.0},
            {(0, 0): -1.0, (1, 0): 1.0},
        ),
        # M6 = (A10-A00)(B00+B01): C11 += M6
        (
            {(1, 0): 1.0, (0, 0): -1.0},
            {(0, 0): 1.0, (0, 1): 1.0},
            {(1, 1): 1.0},
        ),
        # M7 = (A01-A11)(B10+B11): C00 += M7
        (
            {(0, 1): 1.0, (1, 1): -1.0},
            {(1, 0): 1.0, (1, 1): 1.0},
            {(0, 0): 1.0},
        ),
    ]

    col = 0
    for block in range(2):
        k_offset = block * 2  # global columns k = k_offset, k_offset+1

        # 7 Strassen terms for the 2x2 subproblem (j=0,1)
        for (u_part, v_part, w_part) in strassen_terms:
            for (i, j_local), coeff in u_part.items():
                U[i * m + j_local][col] += coeff
            for (j_local, k_local), coeff in v_part.items():
                V[j_local * p + (k_offset + k_local)][col] += coeff
            for (k_local, i), coeff in w_part.items():
                W[(k_offset + k_local) * n + i][col] += coeff
            col += 1

        # 4 extra terms for row j=2: A[i,2]*B[2,k] for i in {0,1}, k_local in {0,1}
        for i in range(2):
            for k_local in range(2):
                k_global = k_offset + k_local
                U[i * m + 2][col] = 1.0
                V[2 * p + k_global][col] = 1.0
                W[k_global * n + i][col] = 1.0
                col += 1

    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
