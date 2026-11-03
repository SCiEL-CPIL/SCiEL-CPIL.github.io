"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 89
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for A[i,j] * B[j,k] -> C[i,k]
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add a Strassen term
    # u, v, w are lists of coefficients for local indices
    # local_a_idx, local_b_idx, local_c_idx are lists of global indices
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs, local_a_indices, local_b_indices, local_c_indices):
        nonlocal col
        for idx, coeff in zip(local_a_indices, u_coeffs):
            if coeff != 0:
                U[idx][col] = coeff
        for idx, coeff in zip(local_b_indices, v_coeffs):
            if coeff != 0:
                V[idx][col] = coeff
        for idx, coeff in zip(local_c_indices, w_coeffs):
            if coeff != 0:
                W[idx][col] = coeff
        col += 1

    # Define local indices for the 2x2 block (rows 0,1 of A; rows 0,1 of B; rows 0,1 of C)
    # A local: (0,0)->0, (0,1)->1, (1,0)->5, (1,1)->6
    a_loc = [0, 1, 5, 6]
    # B local: (0,0)->0, (0,1)->1, (1,0)->6, (1,1)->7
    b_loc = [0, 1, 6, 7]
    # C local: (0,0)->0, (0,1)->3, (1,0)->1, (1,1)->4
    c_loc = [0, 3, 1, 4] # Order: c11, c12, c21, c22

    # Strassen 7 terms
    # M1 = (a+d)(e+h) -> C11+1, C22+1
    add_strassen_term([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], a_loc, b_loc, c_loc)
    # M2 = (c+d)e -> C21+1, C22-1
    add_strassen_term([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1], a_loc, b_loc, c_loc)
    # M3 = a(f-h) -> C12+1, C22+1
    add_strassen_term([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1], a_loc, b_loc, c_loc)
    # M4 = d(g-e) -> C11-1, C21+1
    add_strassen_term([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0], a_loc, b_loc, c_loc)
    # M5 = (a+b)h -> C11-1, C12+1
    add_strassen_term([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0], a_loc, b_loc, c_loc)
    # M6 = (c-a)(e+f) -> C22-1
    add_strassen_term([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1], a_loc, b_loc, c_loc)
    # M7 = (b-d)(g+h) -> C11+1
    add_strassen_term([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0], a_loc, b_loc, c_loc)

    # Schoolbook for remaining blocks
    # A12 (2x3) rows 0,1 cols 2,3,4. B21 (3x2) rows 2,3,4 cols 0,1. C11 (2x2) rows 0,1 cols 0,1.
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)

    # A11 (2x2) rows 0,1 cols 0,1. B12 (2x4) rows 0,1 cols 2,3,4,5. C12 (2x4) rows 0,1 cols 2,3,4,5.
    for i in range(2):
        for j in range(2):
            for k in range(2, 6):
                add_schoolbook(i, j, k)

    # A12 (2x3) rows 0,1 cols 2,3,4. B22 (3x4) rows 2,3,4 cols 2,3,4,5. C12 (2x4) rows 0,1 cols 2,3,4,5.
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 6):
                add_schoolbook(i, j, k)

    # A21 (1x2) row 2 cols 0,1. B11 (2x2) rows 0,1 cols 0,1. C21 (1x2) row 2 cols 0,1.
    for i in range(2, 3):
        for j in range(2):
            for k in range(2):
                add_schoolbook(i, j, k)

    # A22 (1x3) row 2 cols 2,3,4. B21 (3x2) rows 2,3,4 cols 0,1. C21 (1x2) row 2 cols 0,1.
    for i in range(2, 3):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)

    # A21 (1x2) row 2 cols 0,1. B12 (2x4) rows 0,1 cols 2,3,4,5. C22 (1x4) row 2 cols 2,3,4,5.
    for i in range(2, 3):
        for j in range(2):
            for k in range(2, 6):
                add_schoolbook(i, j, k)

    # A22 (1x3) row 2 cols 2,3,4. B22 (3x4) rows 2,3,4 cols 2,3,4,5. C22 (1x4) row 2 cols 2,3,4,5.
    for i in range(2, 3):
        for j in range(2, 5):
            for k in range(2, 6):
                add_schoolbook(i, j, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
