import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    U = [[0.0]*rank for _ in range(n*m)]  # 6 x 28
    V = [[0.0]*rank for _ in range(m*p)]  # 15 x 28
    W = [[0.0]*rank for _ in range(p*n)]  # 10 x 28
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] (2x1 times 1x5), rank 10
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0..4
            U[i*m + 0][col] = 1.0
            V[0*p + k][col] = 1.0
            W[k*n + i][col] = 1.0
            col += 1
    
    # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x5), using Strassen for 2x2 blocks + schoolbook for 2x1
    # A_b[i][j'] = A[i][j'+1], j'=0,1
    # B_b[j'][k] = B[j'+1][k], j'=0,1
    
    # Part 2a: 2x2 * 2x2 (cols 0,1 of B_b, i.e., cols 0,1 of B), Strassen
    # Part 2b: 2x2 * 2x2 (cols 2,3 of B_b, i.e., cols 2,3 of B), Strassen
    # Part 2c: 2x2 * 2x1 (col 4 of B_b, i.e., col 4 of B), schoolbook rank 4
    
    def add_strassen_2x2x2x2(k_offset):
        """Add Strassen decomposition for A_b (2x2) * B_b[:, k_offset:k_offset+2] (2x2).
        k_offset is the column offset in B (0, 2).
        Returns number of columns used."""
        nonlocal col
        # Strassen multipliers
        # M1 = (X00+X11)*(Y00+Y11)
        # M2 = (X10+X11)*Y00
        # M3 = X00*(Y01-Y11)
        # M4 = X11*(Y10-Y00)
        # M5 = (X00+X01)*Y11
        # M6 = (X10-X00)*(Y00+Y01)
        # M7 = (X01-X11)*(Y10+Y11)
        
        # X[i][j'] = A[i][j'+1], so U index = i*3 + (j'+1)
        # Y[j'][k'] = B[j'+1][k_offset+k'], so V index = (j'+1)*5 + (k_offset+k')
        # C[i][k'] = W index = (k_offset+k')*2 + i
        
        strassen_terms = [
            # (coeffs for X, coeffs for Y, coeffs for C)
            # M1: (X00+X11) * (Y00+Y11) -> C00 += 1, C11 += 1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2: (X10+X11) * Y00 -> C10 += 1, C11 -= 1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3: X00 * (Y01-Y11) -> C00 -= 1, C01 += 1
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):-1, (0,1):1}),
            # M4: X11 * (Y10-Y00) -> C00 += 1, C10 += 1
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5: (X00+X01) * Y11 -> C00 -= 1, C01 += 1
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6: (X10-X00) * (Y00+Y01) -> C11 += 1
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7: (X01-X11) * (Y10+Y11) -> C00 += 1
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for x_coeffs, y_coeffs, c_coeffs in strassen_terms:
            for (xi, xj), xv in x_coeffs.items():
                U[xi*3 + (xj+1)][col] += xv
            for (yj, yk), yv in y_coeffs.items():
                V[(yj+1)*5 + (k_offset+yk)][col] += yv
            for (ci, ck), cv in c_coeffs.items():
                W[(k_offset+ck)*2 + ci][col] += cv
            col += 1
        return 7
    
    # Part 2a: cols 0,1
    add_strassen_2x2x2x2(0)
    # Part 2b: cols 2,3
    add_strassen_2x2x2x2(2)
    
    # Part 2c: 2x2 * 2x1 (col 4), schoolbook rank 4
    for i in range(2):  # row of A_b
        for j in range(2):  # col of A_b (= row of B_b)
            U[i*3 + (j+1)][col] = 1.0
            V[(j+1)*5 + 4][col] = 1.0
            W[4*2 + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
