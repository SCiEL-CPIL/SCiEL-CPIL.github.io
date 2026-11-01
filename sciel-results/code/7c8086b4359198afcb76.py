"""Block-strassen decomposition for 2x2 x 2x16 matrix multiplication."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 16 columns = 8 blocks of 2 columns each
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7  # 56

    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2x2:
    # For block with local indices i (row of A/C), j (row of A/B = col of A), 
    # k_loc (col of B/C):
    # M1 = (A[i=0,j=0] + A[i=1,j=1]) * (B[j=0,k=0] + B[j=1,k=1])
    # M2 = (A[i=1,j=0] + A[i=1,j=1]) * B[j=0,k=0]
    # M3 = A[i=0,j=0] * (B[j=1,k=0] - B[j=1,k=1])
    # M4 = A[i=1,j=1] * (B[j=0,k=1] - B[j=0,k=0])
    # M5 = (A[i=0,j=0] + A[i=0,j=1]) * B[j=1,k=1]
    # M6 = (A[i=1,j=0] - A[i=0,j=0]) * (B[j=0,k=0] + B[j=0,k=1])
    # M7 = (A[i=0,j=1] - A[i=1,j=1]) * (B[j=1,k=0] + B[j=1,k=1])
    
    # U index: i*m + j  (i is row of A/C, j is col of A = row of B)
    # V index: j*p + k  (j is row of B, k is col of B/C)
    # W index: k*n + i  (k is col of B/C, i is row of A/C)
    
    # Strassen term definitions:
    # Each term is (u_coeffs, v_coeffs, w_coeffs) where:
    # u_coeffs: list of (i, j, coeff) for U
    # v_coeffs: list of (j, k_loc, coeff) for V
    # w_coeffs: list of (k_loc, i, coeff) for W
    # i in {0,1}, j in {0,1}, k_loc in {0,1}
    
    strassen_terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1), (0,1,-1), (1,0,-1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,0,1), (1,0,-1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(1,1,1), (0,1,-1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (1,1,-1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1), (1,0,1), (0,1,-1), (1,1,-1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,0,1), (0,0,-1), (1,1,-1)]),
    ]

    col = 0
    for block in range(num_blocks):
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            for (i, j, coeff_u) in u_coeffs:
                U[i * m + j][col] += coeff_u
            for (j, k_loc, coeff_v) in v_coeffs:
                k_global = block * 2 + k_loc
                V[j * p + k_global][col] += coeff_v
            for (k_loc, i, coeff_w) in w_coeffs:
                k_global = block * 2 + k_loc
                W[k_global * n + i][col] += coeff_w
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
