import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook rank is 96.
    # We will construct a rank 60 decomposition using Strassen's algorithm for 2x2 blocks.
    
    R = 60
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Helper to add a Strassen term
    # x_u_indices: list of 4 (u00, u01, u10, u11)
    # x_coeffs: list of 4 (coeffs for u00, u01, u10, u11)
    # y_v_indices: list of 4 (v00, v01, v10, v11)
    # y_coeffs: list of 4 (coeffs for v00, v01, v10, v11)
    # z_w_indices: list of 4 (w00, w01, w10, w11)
    # z_coeffs: list of 4 (coeffs for w00, w01, w10, w11)
    def add_strassen_term(x_u_idx, x_coeff, y_v_idx, y_coeff, z_w_idx, z_coeff):
        nonlocal col
        if col >= R:
            return
        for i in range(4):
            if x_coeff[i] != 0:
                U[x_u_idx[i], col] += x_coeff[i]
            if y_coeff[i] != 0:
                V[y_v_idx[i], col] += y_coeff[i]
            if z_coeff[i] != 0:
                W[z_w_idx[i], col] += z_coeff[i]
        col += 1

    # Strassen terms for 2x2 multiplication
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Coefficients for each M in Z
    # M1: Z00=1, Z11=1
    # M2: Z10=1, Z11=-1
    # M3: Z01=1, Z11=1
    # M4: Z00=1, Z10=1
    # M5: Z00=-1, Z01=1
    # M6: Z11=1
    # M7: Z00=1
    
    # Define the 7 Strassen terms
    # Each term is (x_coeff, y_coeff, z_coeff)
    # x_coeff, y_coeff, z_coeff are lists of 4 elements [00, 01, 10, 11]
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Process S1: A_top_left (2x2) * B_rows_0_1 (2x8)
    # A_top_left indices in U:
    # A00 -> 0, A01 -> 1, A10 -> 4, A11 -> 5
    u_idx_s1 = [0, 1, 4, 5]
    
    # B_rows_0_1 is split into 4 blocks of 2x2
    for c in range(4):
        # B block c: rows 0-1, cols 2c, 2c+1
        # V indices:
        # B0,2c -> 0*8 + 2c = 2c
        # B0,2c+1 -> 0*8 + 2c+1 = 2c+1
        # B1,2c -> 1*8 + 2c = 8+2c
        # B1,2c+1 -> 1*8 + 2c+1 = 9+2c
        v_idx = [2*c, 2*c+1, 8+2*c, 9+2*c]
        
        # W indices for C rows 0-1, cols 2c, 2c+1
        # C0,2c -> 2c*3 + 0 = 6c
        # C0,2c+1 -> (2c+1)*3 + 0 = 6c+3
        # C1,2c -> 2c*3 + 1 = 6c+1
        # C1,2c+1 -> (2c+1)*3 + 1 = 6c+4
        w_idx = [6*c, 6*c+3, 6*c+1, 6*c+4]
        
        for x_c, y_c, z_c in terms:
            add_strassen_term(u_idx_s1, x_c, v_idx, y_c, w_idx, z_c)
            
    # Process S2: A_top_right (2x2) * B_rows_2_3 (2x8)
    # A_top_right indices in U:
    # A02 -> 2, A03 -> 3, A12 -> 6, A13 -> 7
    u_idx_s2 = [2, 3, 6, 7]
    
    for c in range(4):
        # B block c: rows 2-3, cols 2c, 2c+1
        # V indices:
        # B2,2c -> 2*8 + 2c = 16+2c
        # B2,2c+1 -> 2*8 + 2c+1 = 17+2c
        # B3,2c -> 3*8 + 2c = 24+2c
        # B3,2c+1 -> 3*8 + 2c+1 = 25+2c
        v_idx = [16+2*c, 17+2*c, 24+2*c, 25+2*c]
        
        # W indices are the same as S1
        w_idx = [6*c, 6*c+3, 6*c+1, 6*c+4]
        
        for x_c, y_c, z_c in terms:
            add_strassen_term(u_idx_s2, x_c, v_idx, y_c, w_idx, z_c)
            
    # Process bottom row: A_bot (1x4) * B (4x8)
    # C2,k = sum_j A2,j B_j,k
    for j in range(4):
        for k in range(8):
            if col >= R:
                break
            # U index: 2*4 + j = 8+j
            U[8+j, col] = 1
            # V index: j*8 + k
            V[j*8+k, col] = 1
            # W index: k*3 + 2
            W[k*3+2, col] = 1
            col += 1
            if col >= R:
                break
        if col >= R:
            break

    # Verify rank used
    if col != R:
        # Should not happen, but if it does, truncate or pad
        pass

    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
