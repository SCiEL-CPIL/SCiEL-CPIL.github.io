import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use block decomposition: 3x3 blocks of 2x2 matrices.
    # Rank = 3 * 3 * 3 * 7 = 189.
    rank = 189
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y11+Y01)
    
    # Coefficients for X (U) and Y (V) for each M_j
    # Indices: 00, 01, 10, 11
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    V_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 0, 0, 1], # M4 (Wait, M4 is X11 * (Y01-Y00). So Y01=1, Y00=-1)
        # Correction M4: Y01 - Y00. Coeffs: 0, 1, 0, -1?
        # Let's re-verify Strassen formulas.
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # M1 = (X00+X11)(Y00+Y11) -> X: 1,0,0,1 ; Y: 1,0,0,1
        # M2 = (X10+X11)Y00       -> X: 0,0,1,1 ; Y: 1,0,0,0
        # M3 = X00(Y10-Y11)       -> X: 1,0,0,0 ; Y: 0,0,1,-1
        # M4 = X11(Y01-Y00)       -> X: 0,0,0,1 ; Y: 0,1,0,-1
        # M5 = (X00+X01)Y11       -> X: 1,1,0,0 ; Y: 0,0,0,1
        # M6 = (X10-X00)(Y00+Y10) -> X: -1,0,1,0; Y: 1,0,1,0
        # M7 = (X01-X11)(Y11+Y01) -> X: 0,1,0,-1; Y: 0,1,0,1
        
    ]
    # Corrected V_coeffs
    V_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, 0, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # Coefficients for Z (W) reconstruction
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    W_coeffs = [
        [1, 0, 0, 1],  # M1 contributes to Z00(1), Z11(1)
        [0, 0, 1, -1], # M2 contributes to Z10(1), Z11(-1)
        [0, 1, 0, 1],  # M3 contributes to Z01(1), Z11(1)
        [1, 0, 1, 0],  # M4 contributes to Z00(1), Z10(1)
        [-1, 1, 0, 0], # M5 contributes to Z00(-1), Z01(1)
        [0, 0, 0, 1],  # M6 contributes to Z11(1)
        [1, 0, 0, 0]   # M7 contributes to Z00(1)
    ]
    # Note: W_coeffs[j] gives coeffs for Z00, Z01, Z10, Z11 for product M_j.
    
    col = 0
    for r in range(3):
        for c in range(3):
            for k in range(3):
                for j in range(7):
                    # U: Input A block rk
                    # Global indices for A[2r+i, 2k+l]
                    # i, l in 0..1
                    # Map local (i,l) to global (row, col)
                    # row = 2*r + i, col = 2*k + l
                    # Index in U (flattened A) = row * 6 + col
                    
                    for i in range(2):
                        for l in range(2):
                            coeff = U_coeffs[j][i*2 + l] # Assuming 00, 01, 10, 11 order
                            # Wait, my U_coeffs list is [00, 01, 10, 11]?
                            # M1: 1,0,0,1 -> 00:1, 01:0, 10:0, 11:1. Correct.
                            if coeff != 0:
                                global_row = 2*r + i
                                global_col = 2*k + l
                                U[global_row * 6 + global_col, col] = coeff
                                
                    # V: Input B block ck
                    # Global indices for B[2c+i, 2k+l]
                    # row = 2*c + i, col = 2*k + l
                    # Index in V (flattened B) = row * 6 + col
                    for i in range(2):
                        for l in range(2):
                            coeff = V_coeffs[j][i*2 + l]
                            if coeff != 0:
                                global_row = 2*c + i
                                global_col = 2*k + l
                                V[global_row * 6 + global_col, col] = coeff
                                
                    # W: Output C block rc
                    # Global indices for C[2r+i, 2c+l]
                    # row = 2*r + i, col = 2*c + l
                    # Index in W (flattened C) = row * 6 + col
                    # W_coeffs[j] is [Z00, Z01, Z10, Z11]
                    for i in range(2):
                        for l in range(2):
                            coeff = W_coeffs[j][i*2 + l]
                            if coeff != 0:
                                global_row = 2*r + i
                                global_col = 2*c + l
                                W[global_row * 6 + global_col, col] = coeff
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
