import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen terms definition
    # X, Y are 2x2.
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y10-Y11)
    # M4: X11(Y01-Y00)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    
    # Coefficients for X (U) and Y (V)
    # Indices in 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    x_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    y_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, -1, 1, 0], # M3
        [0, 1, 0, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Reconstruction coefficients for Z (W)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M5 - M7
    # Z10 = M2 + M3
    # Z11 = M1 - M3 + M6 - M2
    # Indices in Z: 00, 01, 10, 11 -> 0, 1, 2, 3
    z_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 0, 1, -1], # M3
        [1, 0, 0, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, -1, 0, 0]  # M7
    ]
    
    # Define the 4 blocks
    # Block 0: A_L B_T1
    # A_L rows 0,1; cols 0,1. B_T1 rows 0,1; cols 0,1.
    # Output C rows 0,1; cols 0,1.
    # Block 1: A_L B_T2
    # A_L rows 0,1; cols 0,1. B_T2 rows 0,1; cols 2,3.
    # Output C rows 0,1; cols 2,3.
    # Block 2: A_R B_B1
    # A_R rows 0,1; cols 2,3. B_B1 rows 2,3; cols 0,1.
    # Output C rows 0,1; cols 0,1.
    # Block 3: A_R B_B2
    # A_R rows 0,1; cols 2,3. B_B2 rows 2,3; cols 2,3.
    # Output C rows 0,1; cols 2,3.
    
    blocks = [
        {'a_r': [0, 1], 'a_c': [0, 1], 'b_r': [0, 1], 'b_c': [0, 1], 'c_r': [0, 1], 'c_c': [0, 1]},
        {'a_r': [0, 1], 'a_c': [0, 1], 'b_r': [0, 1], 'b_c': [2, 3], 'c_r': [0, 1], 'c_c': [2, 3]},
        {'a_r': [0, 1], 'a_c': [2, 3], 'b_r': [2, 3], 'b_c': [0, 1], 'c_r': [0, 1], 'c_c': [0, 1]},
        {'a_r': [0, 1], 'a_c': [2, 3], 'b_r': [2, 3], 'b_c': [2, 3], 'c_r': [0, 1], 'c_c': [2, 3]}
    ]
    
    col = 0
    for b in blocks:
        for t in range(7):
            # Update U
            for li in range(2):
                for lj in range(2):
                    coeff = x_coeffs[t][li * 2 + lj]
                    if coeff != 0:
                        global_i = b['a_r'][li]
                        global_j = b['a_c'][lj]
                        U[global_i * m + global_j][col] += coeff
            
            # Update V
            for li in range(2):
                for lj in range(2):
                    coeff = y_coeffs[t][li * 2 + lj]
                    if coeff != 0:
                        global_k = b['b_r'][li]
                        global_l = b['b_c'][lj]
                        V[global_k * p + global_l][col] += coeff
            
            # Update W
            for li in range(2):
                for lj in range(2):
                    coeff = z_coeffs[t][li * 2 + lj]
                    if coeff != 0:
                        global_i = b['c_r'][li]
                        global_l = b['c_c'][lj]
                        # W index: k * n + i. Here k is col of C (l), i is row of C (i).
                        # Wait, W[k*n+i]. k is 0..3, i is 0..1.
                        # C[i, l]. i is row, l is col.
                        # So W index is l * n + i.
                        W[global_l * n + global_i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
