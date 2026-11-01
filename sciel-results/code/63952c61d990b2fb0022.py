"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # Indexing convention:
    #   U has shape (n*m, rank) = (4, 7); row i*m + j
    #   V has shape (m*p, rank) = (4, 7); row j*p + k
    #   W has shape (p*n, rank) = (4, 7); row k*n + i  (official trace convention W[k*n+i, r])
    #
    # The multiplication tensor T[i,j,k] satisfies:
    #   (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition requires:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Strassen's 7 intermediate products:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # Output:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # For each intermediate r, we need U entries (coefficients of A elements),
    # V entries (coefficients of B elements), and W entries (coefficients in output).
    #
    # n=m=p=2
    # A indices: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    #   U row index: i*m+j -> A00:0, A01:1, A10:2, A11:3
    # B indices: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
    #   V row index: j*p+k -> B00:0, B01:1, B10:2, B11:3
    # C indices: C[0,0]=C00, C[0,1]=C01, C[1,0]=C10, C[1,1]=C11
    #   W row index: k*n+i -> C00: k=0,i=0 -> 0; C01: k=1,i=0 -> 2; C10: k=0,i=1 -> 1; C11: k=1,i=1 -> 3

    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Each entry is (u_coeffs, v_coeffs, w_coeffs) where each is a dict from index to coeff
    # u_coeffs: {0: A00, 1: A01, 2: A10, 3: A11}
    # v_coeffs: {0: B00, 1: B01, 2: B10, 3: B11}
    # w_coeffs: {0: C00, 1: C10, 2: C01, 3: C11}  (using W[k*n+i] indexing)

    strassen = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 1}),
        # M2 = (A10 + A11) * B00
        ({2: 1, 3: 1}, {0: 1}, {0: 0, 1: 1, 2: 0, 3: -1}),
        # M3 = A00 * (B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {0: 0, 1: 0, 2: 1, 3: 1}),
        # M4 = A11 * (B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1, 2: 0, 3: 0}),
        # M5 = (A00 + A01) * B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 0, 2: 1, 3: 0}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 0, 1: 0, 2: 0, 3: 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 0}),
    ]

    for r, (u_c, v_c, w_c) in enumerate(strassen):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
