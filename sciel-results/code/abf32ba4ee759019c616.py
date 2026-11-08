"""Matrix multiplication 2x4 by 4x4: rank-minimal decomposition."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 4, 4

    # Rank-28 Strassen-based decomposition for 2x4 x 4x4
    # A (2x4) split into A11 (2x2), A12 (2x2)
    # B (4x4) split into B11, B12, B21, B22 (each 2x2)
    # C (2x4) = A*B, C11 = A11*B11 + A12*B21, C12 = A11*B12 + A12*B22
    #
    # Strassen mults (4 non-zero):
    # M1 = A11 * (B11+B22)
    # M3 = A11 * (B12-B22)
    # M5 = (A11+A12) * B22
    # M7 = A12 * (B21+B22)
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    #
    # Each 2x2 mult uses Strassen rank-7. Total rank = 4*7 = 28.
    #
    # Index conventions:
    # U row index: i*m + j where i in {0,1}, j in {0,1,2,3}
    #   A11: i in {0,1}, j in {0,1} -> rows 0,1,2,3 (i*m+j = i*4+j)
    #   A12: i in {0,1}, j in {2,3} -> rows 2,3,6,7
    # V row index: j*p + k where j in {0,1,2,3}, k in {0,1,2,3}
    #   B11: j in {0,1}, k in {0,1} -> rows 0,1,4,5
    #   B12: j in {0,1}, k in {2,3} -> rows 2,3,6,7
    #   B21: j in {2,3}, k in {0,1} -> rows 8,9,12,13
    #   B22: j in {2,3}, k in {2,3} -> rows 10,11,14,15
    # W row index: k*n + i where k in {0,1,2,3}, i in {0,1}
    #   C11: k in {0,1}, i in {0,1} -> rows 0,1,2,3
    #   C12: k in {2,3}, i in {0,1} -> rows 4,5,6,7

    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28

    # Strassen 2x2 decomposition:
    # For P = X * Y where X,Y are 2x2:
    # X indices: X[0]=X11, X[1]=X12, X[2]=X21, X[3]=X22
    # Y indices: Y[0]=Y11, Y[1]=Y12, Y[2]=Y21, Y[3]=Y22
    # P indices: P[0]=P11, P[1]=P12, P[2]=P21, P[3]=P22
    #
    # 7 scalar multiplications:
    # P1 = (X11+X22)*(Y11+Y22)
    # P2 = (X21+X22)*Y11
    # P3 = X11*(Y12-Y22)
    # P4 = X22*(Y21-Y11)
    # P5 = (X11+X12)*Y22
    # P6 = (X21-X11)*(Y11+Y12)
    # P7 = (X12-X22)*(Y21+Y22)
    #
    # Result:
    # P11 = P1 + P4 - P5 + P7
    # P12 = P3 + P5
    # P21 = P2 + P4
    # P22 = P1 - P2 + P3 + P6

    # Helper: for a 2x2 Strassen decomposition, given the 4 X-row indices,
    # 4 Y-row indices, 4 P-row indices, and a base column offset,
    # fill in the U, V, W contributions.
    def strassen_2x2(x_rows, y_rows, p_rows, base_col):
        """
        x_rows: list of 4 U-row indices for X (X11, X12, X21, X22)
        y_rows: list of 4 V-row indices for Y (Y11, Y12, Y21, Y22)
        p_rows: list of 4 W-row indices for P (P11, P12, P21, P22)
        base_col: starting column in U,V,W
        """
        # Each of the 7 Strassen terms contributes to U, V, W
        # Term t: U coefficient * V coefficient * W coefficient
        # We need to express sums/differences in the factors.
        
        # P1 = (X11+X22)*(Y11+Y22): coeff +1 in P11, +1 in P22
        # This requires splitting into separate rank-1 terms because
        # (a+b)*(c+d) = ac + ad + bc + bd, but Strassen uses one multiply
        # of the sums. In tensor decomposition, (X11+X22)*(Y11+Y22) is
        # a single rank-1 term where U has 1 at X11 and X22 rows,
        # V has 1 at Y11 and Y22 rows, W has 1 at P11 and P22 rows.
        
        # Strassen term 1: (X11+X22) * (Y11+Y22)
        col = base_col
        for r in [x_rows[0], x_rows[3]]:  # X11, X22
            U[r][col] += 1
        for r in [y_rows[0], y_rows[3]]:  # Y11, Y22
            V[r][col] += 1
        W[p_rows[0]][col] += 1   # P11
        W[p_rows[3]][col] += 1   # P22
        col += 1
        
        # Strassen term 2: (X21+X22) * Y11
        for r in [x_rows[2], x_rows[3]]:  # X21, X22
            U[r][col] += 1
        V[y_rows[0]][col] += 1  # Y11
        W[p_rows[2]][col] += 1  # P21
        col += 1
        
        # Strassen term 3: X11 * (Y12-Y22)
        U[x_rows[0]][col] += 1  # X11
        V[y_rows[1]][col] += 1  # Y12
        V[y_rows[3]][col] -= 1  # -Y22
        W[p_rows[1]][col] += 1  # P12
        col += 1
        
        # Strassen term 4: X22 * (Y21-Y11)
        U[x_rows[3]][col] += 1  # X22
        V[y_rows[2]][col] += 1  # Y21
        V[y_rows[0]][col] -= 1  # -Y11
        W[p_rows[0]][col] += 1  # P11
        W[p_rows[2]][col] += 1  # P21
        col += 1
        
        # Strassen term 5: (X11+X12) * Y22
        for r in [x_rows[0], x_rows[1]]:  # X11, X12
            U[r][col] += 1
        V[y_rows[3]][col] += 1  # Y22
        W[p_rows[0]][col] -= 1  # -P11
        W[p_rows[1]][col] += 1  # +P12
        col += 1
        
        # Strassen term 6: (X21-X11) * (Y11+Y12)
        U[x_rows[2]][col] += 1  # X21
        U[x_rows[0]][col] -= 1  # -X11
        for r in [y_rows[0], y_rows[1]]:  # Y11, Y12
            V[r][col] += 1
        W[p_rows[3]][col] += 1  # P22
        col += 1
        
        # Strassen term 7: (X12-X22) * (Y21+Y22)
        U[x_rows[1]][col] += 1  # X12
        U[x_rows[3]][col] -= 1  # -X22
        for r in [y_rows[2], y_rows[3]]:  # Y21, Y22
            V[r][col] += 1
        W[p_rows[0]][col] += 1  # +P11
        W[p_rows[3]][col] += 1  # +P22
        col += 1
        
        return col  # next available column

    # Now define the 4 Strassen multiplications:
    # A11 rows in U: i*4+j for i in {0,1}, j in {0,1}
    # A11: X11=U[0], X12=U[1], X21=U[4], X22=U[5]
    # A12 rows in U: i*4+j for i in {0,1}, j in {2,3}
    # A12: X11=U[2], X12=U[3], X21=U[6], X22=U[7]
    
    # B11: Y11=V[0], Y12=V[1], Y21=V[4], Y22=V[5]
    # B12: Y11=V[2], Y12=V[3], Y21=V[6], Y22=V[7]
    # B21: Y11=V[8], Y12=V[9], Y21=V[12], Y22=V[13]
    # B22: Y11=V[10], Y12=V[11], Y21=V[14], Y22=V[15]
    
    # C11: P11=W[0], P12=W[1], P21=W[2], P22=W[3]
    # C12: P11=W[4], P12=W[5], P21=W[6], P22=W[7]
    
    # M1 = A11 * (B11+B22)
    # X = A11: [0, 1, 4, 5]
    # Y = B11+B22: Y11=V[0]+V[10], Y12=V[1]+V[11], Y21=V[4]+V[14], Y22=V[5]+V[15]
    # This is NOT a simple 2x2 Strassen because Y is a sum of two 2x2 matrices.
    # We need to handle this differently.
    
    # Actually, in the tensor decomposition, (B11+B22) means the Y factor
    # has entries at both B11 and B22 positions. So:
    # Y11 = V[0] + V[10], Y12 = V[1] + V[11], Y21 = V[4] + V[14], Y22 = V[5] + V[15]
    #
    # In the Strassen decomposition, when we write V[y_rows[0]] for Y11,
    # we actually need V at both index 0 and 10.
    #
    # Let me redefine: instead of passing single row indices, pass lists of
    # row indices for each of the 4 Y positions.
    
    # Redo with generalized approach:
    # Clear U, V, W
    for r in range(len(U)):
        for c in range(rank):
            U[r][c] = 0
    for r in range(len(V)):
        for c in range(rank):
            V[r][c] = 0
    for r in range(len(W)):
        for c in range(rank):
            W[r][c] = 0

    def strassen_2x2_general(x_rows, y_rows, p_rows, base_col):
        """
        x_rows: list of 4 U-row indices for X (X11, X12, X21, X22)
        y_rows: list of 4 lists of V-row indices for Y (Y11, Y12, Y21, Y22)
                 Each y_rows[t] is a list of row indices with coefficient +1
        p_rows: list of 4 W-row indices for P (P11, P12, P21, P22)
        base_col: starting column
        """
        col = base_col
        
        # Term 1: (X11+X22) * (Y11+Y22)
        for r in [x_rows[0], x_rows[3]]:
            U[r][col] += 1
        for r in y_rows[0]:
            V[r][col] += 1
        for r in y_rows[3]:
            V[r][col] += 1
        W[p_rows[0]][col] += 1
        W[p_rows[3]][col] += 1
        col += 1
        
        # Term 2: (X21+X22) * Y11
        for r in [x_rows[2], x_rows[3]]:
            U[r][col] += 1
        for r in y_rows[0]:
            V[r][col] += 1
        W[p_rows[2]][col] += 1
        col += 1
        
        # Term 3: X11 * (Y12-Y22)
        U[x_rows[0]][col] += 1
        for r in y_rows[1]:
            V[r][col] += 1
        for r in y_rows[3]:
            V[r][col] -= 1
        W[p_rows[1]][col] += 1
        col += 1
        
        # Term 4: X22 * (Y21-Y11)
        U[x_rows[3]][col] += 1
        for r in y_rows[2]:
            V[r][col] += 1
        for r in y_rows[0]:
            V[r][col] -= 1
        W[p_rows[0]][col] += 1
        W[p_rows[2]][col] += 1
        col += 1
        
        # Term 5: (X11+X12) * Y22
        for r in [x_rows[0], x_rows[1]]:
            U[r][col] += 1
        for r in y_rows[3]:
            V[r][col] += 1
        W[p_rows[0]][col] -= 1
        W[p_rows[1]][col] += 1
        col += 1
        
        # Term 6: (X21-X11) * (Y11+Y12)
        U[x_rows[2]][col] += 1
        U[x_rows[0]][col] -= 1
        for r in y_rows[0]:
            V[r][col] += 1
        for r in y_rows[1]:
            V[r][col] += 1
        W[p_rows[3]][col] += 1
        col += 1
        
        # Term 7: (X12-X22) * (Y21+Y22)
        U[x_rows[1]][col] += 1
        U[x_rows[3]][col] -= 1
        for r in y_rows[2]:
            V[r][col] += 1
        for r in y_rows[3]:
            V[r][col] += 1
        W[p_rows[0]][col] += 1
        W[p_rows[3]][col] += 1
        col += 1
        
        return col

    # A11 U-rows: [0, 1, 4, 5] (X11, X12, X21, X22)
    # A12 U-rows: [2, 3, 6, 7]
    # A11+A12 U-rows: [0,1,2,3,4,5,6,7] with +1 on all... 
    # Wait, A11+A12 means X11=A11[0]+A12[0], etc.
    # X11 = U[0] + U[2], X12 = U[1] + U[3], X21 = U[4] + U[6], X22 = U[5] + U[7]
    # So for A11+A12, x_rows should be lists too. Let me generalize further.
    
    # Clear again
    for r in range(len(U)):
        for c in range(rank):
            U[r][c] = 0
    for r in range(len(V)):
        for c in range(rank):
            V[r][c] = 0
    for r in range(len(W)):
        for c in range(rank):
            W[r][c] = 0

    def strassen_2x2_full(x_rows, y_rows, p_rows, base_col):
        """
        x_rows: list of 4 lists of (row, coeff) for X positions (X11, X12, X21, X22)
        y_rows: list of 4 lists of (row, coeff) for Y positions (Y11, Y12, Y21, Y22)
        p_rows: list of 4 W-row indices for P (P11, P12, P21, P22)
        base_col: starting column
        """
        col = base_col
        
        def add_U(rows_coeffs, coeff_mult=1):
            for (r, c) in rows_coeffs:
                U[r][col] += c * coeff_mult
        
        def add_V(rows_coeffs, coeff_mult=1):
            for (r, c) in rows_coeffs:
                V[r][col] += c * coeff_mult
        
        # Term 1: (X11+X22) * (Y11+Y22)
        add_U(x_rows[0])
        add_U(x_rows[3])
        add_V(y_rows[0])
        add_V(y_rows[3])
        W[p_rows[0]][col] += 1
        W[p_rows[3]][col] += 1
        col += 1
        
        # Term 2: (X21+X22) * Y11
        add_U(x_rows[2])
        add_U(x_rows[3])
        add_V(y_rows[0])
        W[p_rows[2]][col] += 1
        col += 1
        
        # Term 3: X11 * (Y12-Y22)
        add_U(x_rows[0])
        add_V(y_rows[1])
        add_V(y_rows[3], -1)
        W[p_rows[1]][col] += 1
        col += 1
        
        # Term 4: X22 * (Y21-Y11)
        add_U(x_rows[3])
        add_V(y_rows[2])
        add_V(y_rows[0], -1)
        W[p_rows[0]][col] += 1
        W[p_rows[2]][col] += 1
        col += 1
        
        # Term 5: (X11+X12) * Y22
        add_U(x_rows[0])
        add_U(x_rows[1])
        add_V(y_rows[3])
        W[p_rows[0]][col] -= 1
        W[p_rows[1]][col] += 1
        col += 1
        
        # Term 6: (X21-X11) * (Y11+Y12)
        add_U(x_rows[2])
        add_U(x_rows[0], -1)
        add_V(y_rows[0])
        add_V(y_rows[1])
        W[p_rows[3]][col] += 1
        col += 1
        
        # Term 7: (X12-X22) * (Y21+Y22)
        add_U(x_rows[1])
        add_U(x_rows[3], -1)
        add_V(y_rows[2])
        add_V(y_rows[3])
        W[p_rows[0]][col] += 1
        W[p_rows[3]][col] += 1
        col += 1
        
        return col

    # Define X factors as lists of (row, coeff):
    # A11: X11=[(0,1)], X12=[(1,1)], X21=[(4,1)], X22=[(5,1)]
    A11_x = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    # A12: X11=[(2,1)], X12=[(3,1)], X21=[(6,1)], X22=[(7,1)]
    A12_x = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    # A11+A12: X11=[(0,1),(2,1)], X12=[(1,1),(3,1)], X21=[(4,1),(6,1)], X22=[(5,1),(7,1)]
    A11A12_x = [[(0,1),(2,1)], [(1,1),(3,1)], [(4,1),(6,1)], [(5,1),(7,1)]]
    
    # Define Y factors:
    # B11: Y11=[(0,1)], Y12=[(1,1)], Y21=[(4,1)], Y22=[(5,1)]
    B11_y = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    # B12: Y11=[(2,1)], Y12=[(3,1)], Y21=[(6,1)], Y22=[(7,1)]
    B12_y = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    # B21: Y11=[(8,1)], Y12=[(9,1)], Y21=[(12,1)], Y22=[(13,1)]
    B21_y = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    # B22: Y11=[(10,1)], Y12=[(11,1)], Y21=[(14,1)], Y22=[(15,1)]
    B22_y = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    
    # B11+B22: Y11=[(0,1),(10,1)], Y12=[(1,1),(11,1)], Y21=[(4,1),(14,1)], Y22=[(5,1),(15,1)]
    B11B22_y = [[(0,1),(10,1)], [(1,1),(11,1)], [(4,1),(14,1)], [(5,1),(15,1)]]
    # B12-B22: Y11=[(2,1),(-10,1)] -> [(2,1),(10,-1)], etc.
    B12mB22_y = [[(2,1),(10,-1)], [(3,1),(11,-1)], [(6,1),(14,-1)], [(7,1),(15,-1)]]
    # B21+B22: Y11=[(8,1),(10,1)], Y12=[(9,1),(11,1)], Y21=[(12,1),(14,1)], Y22=[(13,1),(15,1)]
    B21B22_y = [[(8,1),(10,1)], [(9,1),(11,1)], [(12,1),(14,1)], [(13,1),(15,1)]]
    
    # P rows: W[k*n+i] = W[k*2+i]
    # C11: k∈{0,1}, i∈{0,1}
    # P11=C[0,0]=W[0], P12=C[0,1]=W[2], P21=C[1,0]=W[1], P22=C[1,1]=W[3]
    C11_p = [0, 2, 1, 3]
    # C12: k∈{2,3}, i∈{0,1}
    # P11=C[0,2]=W[4], P12=C[0,3]=W[6], P21=C[1,2]=W[5], P22=C[1,3]=W[7]
    C12_p = [4, 6, 5, 7]
    
    # M1 = A11 * (B11+B22) -> contributes to C11 with +1
    col = strassen_2x2_full(A11_x, B11B22_y, C11_p, 0)
    
    # M3 = A11 * (B12-B22) -> contributes to C12 with +1
    col = strassen_2x2_full(A11_x, B12mB22_y, C12_p, col)
    
    # M5 = (A11+A12) * B22 -> contributes to C11 with -1, C12 with +1
    # Need to handle the sign. The Strassen decomposition gives P = X*Y.
    # For C11: we need -M5, so W contributions should be negated.
    # For C12: we need +M5.
    # But M5 is a single set of 7 rank-1 terms. We can't split it.
    # 
    # Actually, we need M5 to contribute to BOTH C11 (with -1) and C12 (with +1).
    # This means each of the 7 rank-1 terms of M5 must have W entries in both
    # C11 and C12 rows with appropriate signs.
    #
    # Let me modify: for M5, the W factor for each Strassen term needs to
    # include both C11 and C12 contributions.
    #
    # Strassen term results for P = M5:
    # P11 = T1 + T4 - T5 + T7
    # P12 = T3 + T5
    # P21 = T2 + T4
    # P22 = T1 - T2 + T3 + T6
    #
    # C11 contribution: -M5, so C11 gets -(P11,P12,P21,P22)
    # C12 contribution: +M5, so C12 gets +(P11,P12,P21,P22)
    #
    # For each term Tt, the W row gets contributions to both C11 and C12.
    # I need a custom version that writes to two P blocks.
    
    # Let me handle M5 and M7 specially with dual P targets.
    
    # M5 = (A11+A12) * B22
    # C11 gets -M5, C12 gets +M5
    x5 = A11A12_x
    y5 = B22_y
    col5 = col
    
    # Term 1: (X11+X22)*(Y11+Y22) -> P11 and P22
    for (r, c) in x5[0]:
        U[r][col5] += c
    for (r, c) in x5[3]:
        U[r][col5] += c
    for (r, c) in y5[0]:
        V[r][col5] += c
    for (r, c) in y5[3]:
        V[r][col5] += c
    W[C11_p[0]][col5] += -1  # -P11 for C11
    W[C11_p[3]][col5] += -1  # -P22 for C11
    W[C12_p[0]][col5] += 1   # +P11 for C12
    W[C12_p[3]][col5] += 1   # +P22 for C12
    col5 += 1
    
    # Term 2: (X21+X22)*Y11 -> P21
    for (r, c) in x5[2]:
        U[r][col5] += c
    for (r, c) in x5[3]:
        U[r][col5] += c
    for (r, c) in y5[0]:
        V[r][col5] += c
    W[C11_p[2]][col5] += -1  # -P21 for C11
    W[C12_p[2]][col5] += 1   # +P21 for C12
    col5 += 1
    
    # Term 3: X11*(Y12-Y22) -> P12
    for (r, c) in x5[0]:
        U[r][col5] += c
    for (r, c) in y5[1]:
        V[r][col5] += c
    for (r, c) in y5[3]:
        V[r][col5] -= c
    W[C11_p[1]][col5] += -1  # -P12 for C11
    W[C12_p[1]][col5] += 1   # +P12 for C12
    col5 += 1
    
    # Term 4: X22*(Y21-Y11) -> P11 and P21
    for (r, c) in x5[3]:
        U[r][col5] += c
    for (r, c) in y5[2]:
        V[r][col5] += c
    for (r, c) in y5[0]:
        V[r][col5] -= c
    W[C11_p[0]][col5] += -1  # -P11 for C11
    W[C11_p[2]][col5] += -1  # -P21 for C11
    W[C12_p[0]][col5] += 1   # +P11 for C12
    W[C12_p[2]][col5] += 1   # +P21 for C12
    col5 += 1
    
    # Term 5: (X11+X12)*Y22 -> -P11 and P12
    for (r, c) in x5[0]:
        U[r][col5] += c
    for (r, c) in x5[1]:
        U[r][col5] += c
    for (r, c) in y5[3]:
        V[r][col5] += c
    W[C11_p[0]][col5] += 1   # -(-P11) = +P11 for C11
    W[C11_p[1]][col5] += -1  # -(P12) = -P12 for C11
    W[C12_p[0]][col5] += -1  # +(-P11) = -P11 for C12
    W[C12_p[1]][col5] += 1   # +(P12) = +P12 for C12
    col5 += 1
    
    # Term 6: (X21-X11)*(Y11+Y12) -> P22
    for (r, c) in x5[2]:
        U[r][col5] += c
    for (r, c) in x5[0]:
        U[r][col5] -= c
    for (r, c) in y5[0]:
        V[r][col5] += c
    for (r, c) in y5[1]:
        V[r][col5] += c
    W[C11_p[3]][col5] += -1  # -P22 for C11
    W[C12_p[3]][col5] += 1   # +P22 for C12
    col5 += 1
    
    # Term 7: (X12-X22)*(Y21+Y22) -> P11 and P22
    for (r, c) in x5[1]:
        U[r][col5] += c
    for (r, c) in x5[3]:
        U[r][col5] -= c
    for (r, c) in y5[2]:
        V[r][col5] += c
    for (r, c) in y5[3]:
        V[r][col5] += c
    W[C11_p[0]][col5] += -1  # -P11 for C11
    W[C11_p[3]][col5] += -1  # -P22 for C11
    W[C12_p[0]][col5] += 1   # +P11 for C12
    W[C12_p[3]][col5] += 1   # +P22 for C12
    col5 += 1
    
    col = col5
    
    # M7 = A12 * (B21+B22) -> contributes to C11 with +1
    col = strassen_2x2_full(A12_x, B21B22_y, C11_p, col)
    
    best_U, best_V, best_W = U, V, W

    # Attempt Strassen-based rank-28 decomposition for 4x4 x 4x4,
    # then restrict to 2x4 x 4x4.
    #
    # The 4x4 matrix C = A * B where A is 4x4, B is 4x4.
    # We split into 2x2 blocks of size 2x2:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Standard Strassen (7 multiplications):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # But we need to handle the fact that our A is 2x4, not 4x4.
    # We pad A to 4x4 by adding two zero rows.
    # Then rows 3,4 of A are zero: A21 = [0,0,0,0], A22 = [0,0,0,0].
    #
    # However, the block structure matters. Our A is 2x4. Let's think of it as:
    # A is 2 rows x 4 cols. Pad to 4x4:
    # A_padded = [[a00, a01, a02, a03],
    #             [a10, a11, a12, a13],
    #             [0,   0,   0,   0  ],
    #             [0,   0,   0,   0  ]]
    #
    # 2x2 blocks of A_padded:
    # A11 = [[a00,a01],[a10,a11]], A12 = [[a02,a03],[a12,a13]]
    # A21 = [[0,0],[0,0]], A22 = [[0,0],[0,0]]
    #
    # B is 4x4 with blocks B11, B12, B21, B22 (each 2x2).
    #
    # With A21=0, A22=0:
    # M1 = (A11 + 0) * (B11 + B22) = A11 * (B11 + B22)
    # M2 = (0 + 0) * B11 = 0
    # M3 = A11 * (B12 - B22)
    # M4 = 0 * (B21 - B11) = 0
    # M5 = (A11 + A12) * B22
    # M6 = (0 - A11) * (B11 + B12) = -A11 * (B11 + B12)
    # M7 = (A12 - 0) * (B21 + B22) = A12 * (B21 + B22)
    #
    # C11 = M1 + 0 - M5 + M7 = A11*(B11+B22) - (A11+A12)*B22 + A12*(B21+B22)
    #     = A11*B11 + A11*B22 - A11*B22 - A12*B22 + A12*B21 + A12*B22
    #     = A11*B11 + A12*B21  ✓ (this is top-left 2x2 of C)
    # C12 = M3 + M5 = A11*(B12-B22) + (A11+A12)*B22
    #     = A11*B12 - A11*B22 + A11*B22 + A12*B22
    #     = A11*B12 + A12*B22  ✓ (top-right 2x2)
    # C21 = 0 + 0 = 0  ✓ (bottom-left, should be 0 since A rows 3,4 are 0)
    # C22 = M1 - 0 + M3 + M6 = A11*(B11+B22) + A11*(B12-B22) - A11*(B11+B12)
    #     = A11*B11 + A11*B22 + A11*B12 - A11*B22 - A11*B11 - A11*B12 = 0  ✓
    #
    # So we only need 4 non-zero multiplications: M1, M3, M5, M7 (M2=M4=0).
    # Wait, M2 and M4 are zero, so we have 4 multiplications.
    # But each multiplication is a 2x2 matrix multiply, which itself needs
    # to be decomposed. The "rank" counts scalar multiplications.
    #
    # Actually, for the tensor decomposition, each of the 7 Strassen
    # multiplications is a full 2x2 x 2x2 matrix multiply, which in the
    # scalar decomposition has rank 8 (schoolbook for 2x2). But Strassen
    # reduces the number of 2x2 multiplies from 8 to 7, not the scalar rank.
    #
    # The scalar rank of 4x4 x 4x4 multiplication: Strassen gives 7 * (rank of 2x2)
    # which is 7*8 = 56 for schoolbook 2x2, but the actual scalar rank from
    # Strassen is 7 * 1? No, that's not right.
    #
    # Actually, Strassen's algorithm for n x n matrices uses 7 recursive calls
    # to (n/2) x (n/2) multiplication. The scalar rank of 4x4 is at most
    # 7 * (scalar rank of 2x2) = 7 * 8 = 56, but that's worse than schoolbook 64...
    # No wait, 4*4*4 = 64 schoolbook, Strassen gives 7*2*2*2 = 56. So Strassen
    # does help for 4x4.
    #
    # But for our 2x4 x 4x4 case, the schoolbook rank is 2*4*4 = 32.
    # Can we do better than 32?
    #
    # With the padding approach, we have 4 non-zero 2x2 multiplies (M1, M3, M5, M7).
    # Each 2x2 multiply has scalar rank 8 (schoolbook). So total = 4*8 = 32.
    # That's the same as schoolbook!
    #
    # Hmm, but we can use Strassen for the 2x2 multiplies too!
    # 2x2 multiplication has rank 7 (Strassen) instead of 8 (schoolbook).
    # So 4 * 7 = 28 scalar multiplications.
    #
    # Let me work this out carefully.
    #
    # Each of M1, M3, M5, M7 is a product of two 2x2 matrices.
    # Using Strassen for each: 7 scalar multiplies per 2x2 product.
    # Total: 4 * 7 = 28.
    #
    # Let me build this explicitly.
    #
    # The 4x4 x 4x4 tensor, restricted to our 2x4 x 4x4 sub-problem,
    # can be decomposed with rank 28.
    #
    # Let me define the indices.
    # A is 2x4: A[i][j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j][k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i][k] for i in {0,1}, k in {0,1,2,3}
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    #
    # Block decomposition (2x2 blocks):
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    #
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x2)
    #
    # The 4 Strassen multiplications (with A21=A22=0):
    # M1 = A11 * (B11+B22)
    # M3 = A11 * (B12-B22)
    # M5 = (A11+A12) * B22
    # M7 = A12 * (B21+B22)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    #
    # Now each Mx is a 2x2 matrix product. Using Strassen for 2x2:
    # For a product P = X * Y where X, Y are 2x2:
    # s1 = X11 + X22, s2 = Y11 + Y22
    # s3 = X21 + X22, s4 = Y11
    # s5 = X11, s6 = Y12 - Y22
    # s7 = X22, s8 = Y21 - Y11
    # s9 = X11 + X12, s10 = Y22
    # s11 = X21 - X11, s12 = Y11 + Y12
    # s13 = X12 - X22, s14 = Y21 + Y22
    #
    # P1 = s1*s2, P2 = s3*s4, P3 = s5*s6, P4 = s7*s8
    # P5 = s9*s10, P6 = s11*s12, P7 = s13*s14
    #
    # Result:
    # P[0][0] = P1 + P4 - P5 + P7
    # P[0][1] = P3 + P5
    # P[1][0] = P2 + P4
    # P[1][1] = P1 - P2 + P3 + P6
    #
    # This is getting very complex to write out explicitly for all 4
    # Strassen multiplications. Let me just verify rank 32 is safe and
    # try to find a rank-28 construction.
    #
    # Actually, let me try a different approach: direct rank-28 decomposition
    # using the structure of the 2x4 x 4x4 tensor.
    #
    # The tensor T[i,j,k] = delta (C[i,k] = sum_j A[i,j]*B[j,k])
    # has rank at most 32 (schoolbook). The question is whether it can be
    # done in less.
    #
    # Lower bound: the matrix of partial derivatives or the slice rank gives
    # a lower bound. For 2x4 x 4x4, the rank is known to be at most 28
    # by the embedding argument, and the lower bound is at least... 
    # Actually, I'm not sure of the exact minimum. Let me just go with
    # the verified schoolbook rank 32.

    # Verify the decomposition
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(2, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition: C[i,k] = sum_r U[i*4+j... wait
    # U is indexed by (i,j) where i∈{0,1}, j∈{0,1,2,3}, row = i*4+j
    # V is indexed by (j,k) where j∈{0,1,2,3}, k∈{0,1,2,3}, row = j*4+k
    # W is indexed by (k,i) where k∈{0,1,2,3}, i∈{0,1}, row = k*2+i
    # C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r]
    
    C_computed = np.zeros((2, 4))
    for i in range(2):
        for k in range(4):
            s = 0.0
            for j in range(4):
                for r in range(rank):
                    s += U_np[i*4+j, r] * V_np[j*4+k, r] * W_np[k*2+i, r]
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook rank-32
        U_sb = [[0]*32 for _ in range(8)]
        V_sb = [[0]*32 for _ in range(16)]
        W_sb = [[0]*32 for _ in range(8)]
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    r = i*16 + j*4 + k
                    U_sb[i*4+j][r] = 1
                    V_sb[j*4+k][r] = 1
                    W_sb[k*2+i][r] = 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
