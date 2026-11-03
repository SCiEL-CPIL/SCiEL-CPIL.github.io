"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2 x 2x2 coefficients
    # M1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    # M2 = (a21+a22)b11 -> contributes to c21
    # M3 = a11(b12-b22) -> contributes to c12
    # M4 = a22(b21-b11) -> contributes to c11, c21
    # M5 = (a11+a12)b22 -> contributes to c11(-1), c12
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    # M7 = (a12-a22)(b21+b22) -> contributes to c11, c22
    
    # Local indices: (row, col) where row,col in {0,1}
    # a11=(0,0), a12=(0,1), a21=(1,0), a22=(1,1)
    # b11=(0,0), b12=(0,1), b21=(1,0), b22=(1,1)
    # c11=(0,0), c12=(0,1), c21=(1,0), c22=(1,1)
    
    strassen = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]
    
    # 4 Strassen blocks covering the full 2x4 x 4x4 multiplication
    # (a_col_off, b_row_off, b_col_off, c_col_off)
    blocks = [
        (0, 0, 0, 0),  # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 0, 0),  # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (2, 2, 2, 2),  # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    ]
    
    col = 0
    for a_col_off, b_row_off, b_col_off, c_col_off in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen:
            # Fill U for A: A[i,j] at index i*m+j
            for (li, lj), coeff in a_coeffs.items():
                global_i = li
                global_j = a_col_off + lj
                U[global_i * m + global_j][col] += coeff
            
            # Fill V for B: B[j,k] at index j*p+k
            for (li, lj), coeff in b_coeffs.items():
                global_j = b_row_off + li
                global_k = b_col_off + lj
                V[global_j * p + global_k][col] += coeff
            
            # Fill W for C: C[i,k] at index k*n+i
            for (li, lj), coeff in c_coeffs.items():
                global_i = li
                global_k = c_col_off + lj
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
