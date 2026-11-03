"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen decomposition on 2x2 blocks of 3x3 matrices
    # Rank = 7 * 3 * 3 * 3 = 189
    
    # Strassen coefficients
    # M_t = X_t Y_t
    # C_block = sum c_t M_t
    
    # x[t][bi][bj], y[t][bj][bk], c[t][bi][bk]
    # t in 1..7, bi, bj, bk in 0..1
    
    x = [[[0]*2 for _ in range(2)] for _ in range(8)]
    y = [[[0]*2 for _ in range(2)] for _ in range(8)]
    c = [[[0]*2 for _ in range(2)] for _ in range(8)]
    
    # M1 = (A00 + A11)(B00 + B11)
    x[1][0][0] = 1; x[1][1][1] = 1
    y[1][0][0] = 1; y[1][1][1] = 1
    c[1][0][0] = 1; c[1][1][1] = 1
    
    # M2 = (A21 + A22)B11  -> Note: Strassen standard uses A21, A22? 
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # My indices: 00, 01, 10, 11.
    # A11->A00, A12->A01, A21->A10, A22->A11.
    
    # M2 = (A10 + A11)B00
    x[2][1][0] = 1; x[2][1][1] = 1
    y[2][0][0] = 1
    
    # M3 = A00(B01 - B11)
    x[3][0][0] = 1
    y[3][0][1] = 1; y[3][1][1] = -1
    
    # M4 = A11(B10 - B00)
    x[4][1][1] = 1
    y[4][1][0] = 1; y[4][0][0] = -1
    
    # M5 = (A00 + A01)B11
    x[5][0][0] = 1; x[5][0][1] = 1
    y[5][1][1] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    x[6][1][0] = 1; x[6][0][0] = -1
    y[6][0][0] = 1; y[6][0][1] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    x[7][0][1] = 1; x[7][1][1] = -1
    y[7][1][0] = 1; y[7][1][1] = 1
    
    # Coefficients for C blocks
    # C00 = M1 + M4 - M5 + M7
    c[1][0][0] = 1; c[4][0][0] = 1; c[5][0][0] = -1; c[7][0][0] = 1
    # C01 = M3 + M5
    c[3][0][1] = 1; c[5][0][1] = 1
    # C10 = M2 + M4
    c[2][1][0] = 1; c[4][1][0] = 1
    # C11 = M1 - M2 + M3 + M6
    c[1][1][1] = 1; c[2][1][1] = -1; c[3][1][1] = 1; c[6][1][1] = 1
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(1, 8):
        for li in range(3):
            for lj in range(3):
                for lk in range(3):
                    # U: I, J. I = bi*3+li, J = bj*3+lj
                    for bi in range(2):
                        for bj in range(2):
                            val = x[t][bi][bj]
                            if val != 0:
                                I = bi * 3 + li
                                J = bj * 3 + lj
                                U[I * m + J][col] = val
                                
                    # V: J, K. J = bj*3+lj, K = bk*3+lk
                    for bj in range(2):
                        for bk in range(2):
                            val = y[t][bj][bk]
                            if val != 0:
                                J = bj * 3 + lj
                                K = bk * 3 + lk
                                V[J * p + K][col] = val
                                
                    # W: K, I. K = bk*3+lk, I = bi*3+li
                    for bi in range(2):
                        for bk in range(2):
                            val = c[t][bi][bk]
                            if val != 0:
                                K = bk * 3 + lk
                                I = bi * 3 + li
                                W[K * n + I][col] = val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
