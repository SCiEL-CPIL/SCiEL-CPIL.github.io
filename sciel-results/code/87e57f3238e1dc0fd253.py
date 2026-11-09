"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # We use a decomposition of rank 22.
    # Split p=4 into two parts of 2.
    # Each part is a 2x3x2 multiplication.
    # 2x3x2 is isomorphic to 2x2x3.
    # We decompose 2x3x2 into a 2x2x2 part (rank 7) and a 2x1x2 outer product (rank 4).
    # Total rank per part = 11.
    # Total rank = 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, col
    # U[u_idx, col] += u_val
    # V[v_idx, col] += v_val
    # W[w_idx, col] += w_val
    
    def add_term(u_idx, v_idx, w_idx, col, u_val, v_val, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Indices for A (2x3):
    # a00, a01, a02
    # a10, a11, a12
    # U indices:
    # 0: a00, 1: a01, 2: a02
    # 3: a10, 4: a11, 5: a12
    
    # Indices for B (3x4):
    # b00, b01, b02, b03
    # b10, b11, b12, b13
    # b20, b21, b22, b23
    # V indices:
    # 0: b00, 1: b01, 2: b02, 3: b03
    # 4: b10, 5: b11, 6: b12, 7: b13
    # 8: b20, 9: b21, 10: b22, 11: b23
    
    # Indices for C (4x2) -> W[k*n+i, r] -> W[k*2+i, r]
    # c00, c01
    # c10, c11
    # c20, c21
    # c30, c31
    # W indices:
    # 0: c00, 1: c01
    # 2: c10, 3: c11
    # 4: c20, 5: c21
    # 6: c30, 7: c31
    
    # Process First Half (columns 0, 1 of B)
    # Y1 (2x2) from B: rows 0,1 of B, cols 0,1.
    # Y1 = [[b00, b10], [b01, b11]] -> V indices 0, 4, 1, 5
    # X1 (2x2) from A: rows 0,1 of A (cols 0,1).
    # X1 = [[a00, a10], [a01, a11]] -> U indices 0, 3, 1, 4
    # Y2 (2x1) from B: row 2 of B, cols 0,1.
    # Y2 = [[b20], [b21]] -> V indices 8, 9
    # X2 (1x2) from A: row 2 of A (cols 0,1).
    # X2 = [[a02, a12]] -> U indices 2, 5
    
    # Strassen for Y1 X1
    # M1 = (Y1[0][0]+Y1[1][1]) * (X1[0][0]+X1[1][1])
    # M2 = (Y1[1][0]+Y1[1][1]) * X1[0][0]
    # M3 = Y1[0][0] * (X1[1][1]-X1[1][0])
    # M4 = Y1[1][1] * (X1[0][1]-X1[0][0])
    # M5 = (Y1[0][0]+Y1[0][1]) * X1[1][1]
    # M6 = (Y1[1][0]-Y1[0][0]) * (X1[0][0]+X1[0][1])
    # M7 = (Y1[0][1]-Y1[1][1]) * (X1[1][0]+X1[1][1])
    
    # Note: The standard Strassen formulas compute Z = Y X.
    # Z[0][0] = M1 - M4 + M5 - M6
    # Z[0][1] = M3 + M5
    # Z[1][0] = M2 + M4
    # Z[1][1] = M1 - M2 + M3 + M7
    
    # We need to map these to U, V, W.
    # Each M is a product of a linear combo of Y (V) and X (U).
    # So U gets the X combo, V gets the Y combo, W gets the contribution to Z.
    
    col = 0
    # M1
    # X part: X1[0][0] + X1[1][1] = a00 + a11 (U: 0, 3)
    # Y part: Y1[0][0] + Y1[1][1] = b00 + b11 (V: 0, 5)
    # Z contribution:
    # Z[0][0] += 1, Z[1][1] += 1
    add_term(0, 0, 0, col, 1, 1, 1) # a00 * b00 -> c00
    add_term(3, 0, 1, col, 1, 1, 1) # a11 * b00 -> c11? No.
    # Wait, the term is (a00+a11)(b00+b11).
    # This expands to a00b00 + a00b11 + a11b00 + a11b11.
    # We are adding a rank 1 term: U_vec * V_vec * W_vec.
    # U_vec has 1 at indices corresponding to a00, a11.
    # V_vec has 1 at indices corresponding to b00, b11.
    # W_vec has 1 at indices corresponding to c00, c11.
    # So:
    U[0][col] += 1; U[3][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[3][col] += 1 # c00, c11
    col += 1
    
    # M2
    # X part: X1[0][0] = a00 (U: 0)
    # Y part: Y1[1][0] + Y1[1][1] = b10 + b11 (V: 4, 5)
    # Z contribution: Z[1][0] += 1 (c10)
    U[0][col] += 1
    V[4][col] += 1; V[5][col] += 1
    W[2][col] += 1 # c10
    col += 1
    
    # M3
    # X part: X1[1][1] - X1[1][0] = a11 - a10 (U: 4, 3)
    # Y part: Y1[0][0] = b00 (V: 0)
    # Z contribution: Z[0][1] += 1 (c01), Z[1][1] += 1 (c11)
    U[4][col] += 1; U[3][col] -= 1
    V[0][col] += 1
    W[1][col] += 1; W[3][col] += 1 # c01, c11
    col += 1
    
    # M4
    # X part: X1[0][1] - X1[0][0] = a01 - a00 (U: 1, 0)
    # Y part: Y1[1][1] = b11 (V: 5)
    # Z contribution: Z[0][0] -= 1 (c00), Z[1][0] += 1 (c10)
    U[1][col] += 1; U[0][col] -= 1
    V[5][col] += 1
    W[0][col] -= 1; W[2][col] += 1 # c00, c10
    col += 1
    
    # M5
    # X part: X1[1][1] = a11 (U: 4)
    # Y part: Y1[0][0] + Y1[0][1] = b00 + b01 (V: 0, 1)
    # Z contribution: Z[0][0] -= 1 (c00), Z[0][1] += 1 (c01)
    U[4][col] += 1
    V[0][col] += 1; V[1][col] += 1
    W[0][col] -= 1; W[1][col] += 1 # c00, c01
    col += 1
    
    # M6
    # X part: X1[0][0] + X1[0][1] = a00 + a01 (U: 0, 1)
    # Y part: Y1[1][0] - Y1[0][0] = b10 - b00 (V: 4, 0)
    # Z contribution: Z[0][0] -= 1 (c00)
    U[0][col] += 1; U[1][col] += 1
    V[4][col] += 1; V[0][col] -= 1
    W[0][col] -= 1 # c00
    col += 1
    
    # M7
    # X part: X1[1][0] + X1[1][1] = a10 + a11 (U: 3, 4)
    # Y part: Y1[0][1] - Y1[1][1] = b01 - b11 (V: 1, 5)
    # Z contribution: Z[1][1] += 1 (c11)
    U[3][col] += 1; U[4][col] += 1
    V[1][col] += 1; V[5][col] -= 1
    W[3][col] += 1 # c11
    col += 1
    
    # Outer Product Y2 X2
    # Y2 = [b20, b21]^T (V: 8, 9)
    # X2 = [a02, a12] (U: 2, 5)
    # Z contribution:
    # c00 += a02*b20
    # c01 += a02*b21
    # c10 += a12*b20
    # c11 += a12*b21
    
    # Term 1: a02 * b20 -> c00
    U[2][col] += 1
    V[8][col] += 1
    W[0][col] += 1
    col += 1
    
    # Term 2: a02 * b21 -> c01
    U[2][col] += 1
    V[9][col] += 1
    W[1][col] += 1
    col += 1
    
    # Term 3: a12 * b20 -> c10
    U[5][col] += 1
    V[8][col] += 1
    W[2][col] += 1
    col += 1
    
    # Term 4: a12 * b21 -> c11
    U[5][col] += 1
    V[9][col] += 1
    W[3][col] += 1
    col += 1
    
    # Process Second Half (columns 2, 3 of B)
    # Y1' (2x2) from B: rows 0,1 of B, cols 2,3.
    # Y1' = [[b02, b12], [b03, b13]] -> V indices 2, 6, 3, 7
    # X1' (2x2) from A: rows 0,1 of A (cols 0,1). SAME X1.
    # X1' = [[a00, a10], [a01, a11]] -> U indices 0, 3, 1, 4
    # Y2' (2x1) from B: row 2 of B, cols 2,3.
    # Y2' = [[b22], [b23]] -> V indices 10, 11
    # X2' (1x2) from A: row 2 of A (cols 0,1). SAME X2.
    # X2' = [[a02, a12]] -> U indices 2, 5
    
    # Strassen for Y1' X1'
    # M1'
    U[0][col] += 1; U[3][col] += 1
    V[2][col] += 1; V[7][col] += 1
    W[4][col] += 1; W[7][col] += 1 # c20, c31
    col += 1
    
    # M2'
    U[0][col] += 1
    V[6][col] += 1; V[7][col] += 1
    W[6][col] += 1 # c30
    col += 1
    
    # M3'
    U[4][col] += 1; U[3][col] -= 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] += 1 # c21, c31
    col += 1
    
    # M4'
    U[1][col] += 1; U[0][col] -= 1
    V[7][col] += 1
    W[4][col] -= 1; W[6][col] += 1 # c20, c30
    col += 1
    
    # M5'
    U[4][col] += 1
    V[2][col] += 1; V[3][col] += 1
    W[4][col] -= 1; W[5][col] += 1 # c20, c21
    col += 1
    
    # M6'
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1; V[2][col] -= 1
    W[4][col] -= 1 # c20
    col += 1
    
    # M7'
    U[3][col] += 1; U[4][col] += 1
    V[3][col] += 1; V[7][col] -= 1
    W[7][col] += 1 # c31
    col += 1
    
    # Outer Product Y2' X2'
    # Y2' = [b22, b23]^T (V: 10, 11)
    # X2' = [a02, a12] (U: 2, 5)
    
    # Term 1: a02 * b22 -> c20
    U[2][col] += 1
    V[10][col] += 1
    W[4][col] += 1
    col += 1
    
    # Term 2: a02 * b23 -> c21
    U[2][col] += 1
    V[11][col] += 1
    W[5][col] += 1
    col += 1
    
    # Term 3: a12 * b22 -> c30
    U[5][col] += 1
    V[10][col] += 1
    W[6][col] += 1
    col += 1
    
    # Term 4: a12 * b23 -> c31
    U[5][col] += 1
    V[11][col] += 1
    W[7][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
