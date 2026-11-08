import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block structure: 3x3 blocks of size 2x2
    block_size = 2
    num_blocks = 3
    
    # Total rank: 9 block positions * 3 sum terms * 7 Strassen terms = 189
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = (a-b)(f+h)
    # M4 = d(h-g)
    # M5 = (a+c)(e-g)
    # M6 = (b+d)(h-e)
    # M7 = (a-c)(f+g)
    
    # Local indices for 2x2 matrices
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [a, b, c, d], V: [e, f, g, h], W: [C11, C12, C21, C22]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2 -> C21
        ([1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, 0]), # M3 -> C12
        ([0, 0, 0, 1], [0, 0, -1, 1], [1, 0, 0, 0]), # M4 -> C11
        ([1, 0, 1, 0], [1, 0, -1, 0], [-1, 0, 0, 0]),# M5 -> C11
        ([0, 1, 0, 1], [0, -1, 0, 1], [0, 0, 0, 1]), # M6 -> C22
        ([1, 0, -1, 0], [0, 1, 1, 0], [1, 0, 0, -1]) # M7 -> C11, C22
    ]
    
    # Note: M7 contributes to C11 (+1) and C22 (-1).
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # Let's re-verify the W coefficients for each term based on the formulas.
    # M1: C11(+1), C22(+1). W: [1, 0, 0, 1]
    # M2: C21(+1). W: [0, 0, 1, 0]
    # M3: C12(+1), C22(-1). W: [0, 1, 0, -1]
    # M4: C11(+1), C21(+1). W: [1, 0, 1, 0]
    # M5: C11(-1), C12(+1). W: [-1, 1, 0, 0]
    # M6: C22(+1). W: [0, 0, 0, 1]
    # M7: C11(+1), C22(-1). W: [1, 0, 0, -1]
    
    # Corrected Strassen Terms List
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
        ([1, 0, -1, 0], [0, 1, 0, 1], [0, 1, 0, -1]),# M3
        ([0, 0, 0, 1], [0, 0, -1, 1], [1, 0, 1, 0]), # M4
        ([1, 0, 1, 0], [1, 0, -1, 0], [-1, 1, 0, 0]),# M5
        ([0, 1, 0, 1], [0, -1, 0, 1], [0, 0, 0, 1]), # M6
        ([1, 0, -1, 0], [0, 1, 1, 0], [1, 0, 0, -1]) # M7
    ]
    
    col_idx = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                # Term A_IK * B_KJ
                for s in range(7):
                    u_coeff, v_coeff, w_coeff = strassen_terms[s]
                    
                    # Fill U
                    for loc_i in range(block_size):
                        for loc_j in range(block_size):
                            a_idx = loc_i * block_size + loc_j
                            if u_coeff[a_idx] != 0:
                                global_i = I * block_size + loc_i
                                global_j = K * block_size + loc_j
                                u_row = global_i * m + global_j
                                U[u_row, col_idx] = u_coeff[a_idx]
                    
                    # Fill V
                    for loc_j in range(block_size):
                        for loc_k in range(block_size):
                            b_idx = loc_j * block_size + loc_k
                            if v_coeff[b_idx] != 0:
                                global_j = K * block_size + loc_j
                                global_k = J * block_size + loc_k
                                v_row = global_j * p + global_k
                                V[v_row, col_idx] = v_coeff[b_idx]
                    
                    # Fill W
                    for loc_k in range(block_size):
                        for loc_i in range(block_size):
                            c_idx = loc_k * block_size + loc_i # W is [k, i]
                            if w_coeff[c_idx] != 0:
                                global_k = J * block_size + loc_k
                                global_i = I * block_size + loc_i
                                w_row = global_k * n + global_i
                                W[w_row, col_idx] = w_coeff[c_idx]
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
