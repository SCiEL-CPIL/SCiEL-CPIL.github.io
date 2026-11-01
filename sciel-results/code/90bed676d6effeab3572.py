"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U vectors for A coefficients (a, b, c, d)
    U_strassen = [
        [1, 0, 0, 1],   # m1
        [0, 0, 1, 1],   # m2
        [1, 0, 0, 0],   # m3
        [0, 0, 0, 1],   # m4
        [1, 1, 0, 0],   # m5
        [-1, 0, 1, 0],  # m6
        [0, 1, 0, -1]   # m7
    ]
    
    # V vectors for B coefficients (e, f, g, h)
    V_strassen = [
        [1, 1, 1, 1],   # m1
        [1, 0, 0, 0],   # m2
        [0, 1, 0, -1],  # m3
        [-1, 0, 1, 0],  # m4
        [0, 0, 0, 1],   # m5
        [1, 1, 0, 0],   # m6
        [0, 0, 1, 1]    # m7
    ]
    
    # W vectors for C coefficients (C00, C10, C01, C11)
    # C00 = m1 + m4 - m5 + m7
    # C10 = m2 + m4
    # C01 = m3 + m5
    # C11 = m1 - m2 + m3 + m6
    W_strassen = [
        [1, 0, 0, 1],   # m1
        [0, 1, 0, -1],  # m2
        [0, 0, 1, 1],   # m3
        [1, 1, 0, 0],   # m4
        [-1, 0, 1, 0],  # m5
        [0, 0, 0, 1],   # m6
        [1, 0, 0, 0]    # m7
    ]
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The matrix A is shared across all 3 blocks, so U is the same for all
    for block in range(3):
        for r in range(7):
            col = block * 7 + r
            for i in range(4):
                U[i][col] = U_strassen[r][i]
                
    # V indices for B[j, k] is j*p + k
    # Block 1 (k=0,1): e=B00(0), f=B01(1), g=B10(6), h=B11(7)
    # Block 2 (k=2,3): e=B02(2), f=B03(3), g=B12(8), h=B13(9)
    # Block 3 (k=4,5): e=B04(4), f=B05(5), g=B14(10), h=B15(11)
    v_indices = [
        [0, 1, 6, 7],
        [2, 3, 8, 9],
        [4, 5, 10, 11]
    ]
    
    # W indices for C[i, k] is k*n + i
    # Block 1 (k=0,1): C00(0), C10(1), C01(2), C11(3)
    # Block 2 (k=2,3): C02(4), C12(5), C03(6), C13(7)
    # Block 3 (k=4,5): C04(8), C14(9), C05(10), C15(11)
    w_indices = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ]
    
    for block in range(3):
        for r in range(7):
            col = block * 7 + r
            for i in range(4):
                V[v_indices[block][i]][col] = V_strassen[r][i]
                W[w_indices[block][i]][col] = W_strassen[r][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
