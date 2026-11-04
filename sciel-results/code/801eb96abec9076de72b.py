"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank is 42 (7 Strassen products * 6 rank per 1x2 by 2x3 block mult)
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen terms definition: (a, b, c, d, sign)
    # A is 2x4, blocks A_ab are 1x2. a in {0,1}, b in {0,1}
    # B is 4x6, blocks B_cd are 2x3. c in {0,1}, d in {0,1}
    # A_ab corresponds to A[a, 2b:2b+2]
    # B_cd corresponds to B[2c:2c+2, 3d:3d+3]
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        [(0, 0, 0, 0, 1), (0, 0, 1, 1, 1), (1, 1, 0, 0, 1), (1, 1, 1, 1, 1)],
        # M2 = (A10 + A11)B00
        [(1, 0, 0, 0, 1), (1, 1, 0, 0, 1)],
        # M3 = A00(B01 - B11)
        [(0, 0, 0, 1, 1), (0, 0, 1, 1, -1)],
        # M4 = A11(B10 - B00)
        [(1, 1, 1, 0, 1), (1, 1, 0, 0, -1)],
        # M5 = (A00 + A01)B11
        [(0, 0, 1, 1, 1), (0, 1, 1, 1, 1)],
        # M6 = (A10 - A00)(B00 + B01)
        [(1, 0, 0, 0, 1), (1, 0, 0, 1, 1), (0, 0, 0, 0, -1), (0, 0, 0, 1, -1)],
        # M7 = (A01 - A11)(B10 + B11)
        [(0, 1, 1, 0, 1), (0, 1, 1, 1, 1), (1, 1, 1, 0, -1), (1, 1, 1, 1, -1)]
    ]
    
    for terms in strassen_terms:
        for (a, b, c, d, sign) in terms:
            # For each term, we decompose the 1x2 by 2x3 multiplication
            # X (1x2) * Y (2x3) = Z (1x3)
            # Z_k = X_0 Y_0k + X_1 Y_1k for k=0,1,2
            # We iterate over the 6 scalar multiplications involved
            for t in range(6):
                if t < 3:
                    # Involves X_0 (row 0 of the 1x2 block) and Y_0t (row 0 of the 2x3 block, col t)
                    u_row = a * m + (2 * b + 0)
                    v_row = (2 * c + 0) * p + (3 * d + t)
                    w_row = (3 * d + t) * n + a
                else:
                    # Involves X_1 (row 1 of the 1x2 block) and Y_1(t-3) (row 1 of the 2x3 block, col t-3)
                    u_row = a * m + (2 * b + 1)
                    v_row = (2 * c + 1) * p + (3 * d + (t - 3))
                    w_row = (3 * d + (t - 3)) * n + a
                
                U[u_row][col_idx] += sign
                V[v_row][col_idx] += sign
                W[w_row][col_idx] += sign
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
