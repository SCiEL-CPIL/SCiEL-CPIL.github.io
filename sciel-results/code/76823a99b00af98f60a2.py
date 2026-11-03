"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Decompose using 2x2 Strassen blocks
    # A is n x m = 2 x 4 = 2 x (2+2)
    # B is m x p = 4 x 6 = (2+2) x (2+2+2)
    # C = A_left @ B_top + A_right @ B_bottom
    # Each 2x2 @ 2x6 is decomposed into three 2x2 @ 2x2 Strassen products
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For X (2x2) @ Y (2x2) = Z (2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # We'll build U, V, W for the full decomposition
    # Each Strassen block contributes 7 rank-1 terms
    # Total: 6 Strassen blocks x 7 = 42 rank-1 terms
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for the 7 multiplications
    # Each Strassen term has: U_part (2x2 submatrix of A), V_part (2x2 submatrix of B), W_part (2x2 submatrix of C)
    
    # Define Strassen factors
    # For 2x2 @ 2x2, the 7 Strassen products are:
    # s1: (X00+X11) * (Y00+Y11) -> contributes to Z00(+), Z11(+)
    # s2: (X10+X11) * Y00       -> contributes to Z10(+)
    # s3: X00 * (Y01-Y11)       -> contributes to Z01(+)
    # s4: X11 * (Y10-Y00)       -> contributes to Z10(+), Z00(+)
    # s5: (X00+X01) * Y11       -> contributes to Z01(+), Z00(-)
    # s6: (X10-X00) * (Y00+Y01) -> contributes to Z11(+)
    # s7: (X01-X11) * (Y10+Y11) -> contributes to Z00(+), Z11(-)
    
    # Strassen U factors (for X matrix, 2x2): each is a row vector in the 2x2 space
    # X00, X01, X10, X11 are the four entries
    # U factors (7 vectors in R^4):
    U_strassen = [
        [1, 0, 0, 1],   # X00+X11
        [0, 0, 1, 1],   # X10+X11
        [1, 0, 0, 0],   # X00
        [0, 0, 0, 1],   # X11
        [1, 1, 0, 0],   # X00+X01
        [-1, 0, 1, 0],  # X10-X00
        [0, 1, 0, -1],  # X01-X11
    ]
    
    # V factors (7 vectors in R^4):
    V_strassen = [
        [1, 1, 0, 0],   # Y00+Y11
        [1, 0, 0, 0],   # Y00
        [0, 1, 0, -1],  # Y01-Y11
        [0, -1, 1, 0],  # Y10-Y00
        [0, 0, 0, 1],   # Y11
        [1, 1, 0, 0],   # Y00+Y01
        [0, 0, 1, 1],   # Y10+Y11
    ]
    
    # W factors (7 vectors in R^4 for the 2x2 output Z):
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    W_strassen = [
        [1, 0, 0, 1],   # M1: Z00+, Z11+
        [0, 0, 1, 0],   # M2: Z10+
        [0, 1, 0, 0],   # M3: Z01+
        [1, 0, 1, 0],   # M4: Z00+, Z10+
        [-1, 1, 0, 0],  # M5: Z00-, Z01+
        [0, 0, 0, 1],   # M6: Z11+
        [1, 0, 0, -1],  # M7: Z00+, Z11-
    ]
    
    # Now, for each of the 6 Strassen blocks:
    # Block (a, c) where a in {0,1} (row-block of A), c in {0,1} (col-block of B rows)
    # And for each of the 3 column-blocks of B (l in {0,1,2})
    
    # A is 2x4. Row i, column j. j = 2*a + b where a in {0,1}, b in {0,1}
    # B is 4x6. Row k, column l. k = 2*c + d where c in {0,1}, d in {0,1}. l = 2*e + f where e in {0,1,2}, f in {0,1}
    
    # For Strassen block (a, c, e):
    # X = A[:, 2a:2a+2] (2x2)
    # Y = B[2c:2c+2, 2e:2e+2] (2x2)
    # Z = C[:, 2e:2e+2] (2x2)
    
    # U index for A entry (i, j): i*m + j = i*4 + j
    # V index for B entry (k, l): k*p + l = k*6 + l
    # W index for C entry (l, i): l*n + i = l*2 + i (trace convention W[k*n+i, r])
    
    col = 0
    for a in range(2):       # row-block of A (0 or 1)
        for c in range(2):   # row-block of B (0 or 1)
            for e in range(3):  # col-block of B (0, 1, or 2)
                for s in range(7):  # Strassen factor
                    # U: for each of the 4 entries of X (2x2 submatrix of A)
                    for bi in range(2):  # row within X (i = bi)
                        for bj in range(2):  # col within X (j = 2*a + bj)
                            u_idx = bi * m + (2 * a + bj)
                            U[u_idx][col] += U_strassen[s][bi * 2 + bj]
                    
                    # V: for each of the 4 entries of Y (2x2 submatrix of B)
                    for bk in range(2):  # row within Y (k = 2*c + bk)
                        for bl in range(2):  # col within Y (l = 2*e + bl)
                            v_idx = (2 * c + bk) * p + (2 * e + bl)
                            V[v_idx][col] += V_strassen[s][bk * 2 + bl]
                    
                    # W: for each of the 4 entries of Z (2x2 submatrix of C)
                    for bi in range(2):  # row within Z (i = bi)
                        for bl in range(2):  # col within Z (l = 2*e + bl)
                            w_idx = (2 * e + bl) * n + bi
                            W[w_idx][col] += W_strassen[s][bi * 2 + bl]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
