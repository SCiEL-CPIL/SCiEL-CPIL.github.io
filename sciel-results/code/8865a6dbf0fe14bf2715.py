import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # P1: (00+11)(00+11) -> C00+=1, C11+=1
    # P2: (10+11)00 -> C10+=1, C11+=1
    # P3: 00(10-11) -> C01+=1, C11-=-1
    # P4: 11(01-00) -> C00-=-1, C01+=1
    # P5: (00+10)11 -> C00+=1, C10+=1
    # P6: (11-00)(00+10) -> C10+=1, C11-=-1
    # P7: (01-11)(11+10) -> C00+=1, C11+=1
    
    # Local indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # A: i*2+j
    # B: j*2+k
    # C: k*2+i
    
    strassen_ops = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: {idx: coeff}, V: {idx: coeff}, W: {idx: coeff}
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}), # P2
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: -1}), # P3
        ({3: 1}, {1: 1, 0: -1}, {0: -1, 1: 1}), # P4
        ({0: 1, 2: 1}, {3: 1}, {0: 1, 2: 1}), # P5
        ({3: 1, 0: -1}, {0: 1, 2: 1}, {2: 1, 3: -1}), # P6
        ({1: 1, 3: -1}, {3: 1, 2: 1}, {0: 1, 3: 1}), # P7
    ]
    
    col = 0
    # Block 1: A_L (j 0,1), B_T (j 0,1)
    # Sub-blocks for k in B
    for k_offset in [0, 2, 4]:
        for op in strassen_ops:
            u_coeffs, v_coeffs, w_coeffs = op
            for u_idx, coeff in u_coeffs.items():
                i, j = divmod(u_idx, 2)
                global_u = i * 4 + j
                U[global_u][col] += coeff
            for v_idx, coeff in v_coeffs.items():
                j, k = divmod(v_idx, 2)
                global_v = j * 6 + (k + k_offset)
                V[global_v][col] += coeff
            for w_idx, coeff in w_coeffs.items():
                k, i = divmod(w_idx, 2)
                global_w = (k + k_offset) * 2 + i
                W[global_w][col] += coeff
            col += 1
            
    # Block 2: A_R (j 2,3), B_B (j 2,3)
    for k_offset in [0, 2, 4]:
        for op in strassen_ops:
            u_coeffs, v_coeffs, w_coeffs = op
            for u_idx, coeff in u_coeffs.items():
                i, j = divmod(u_idx, 2)
                global_u = i * 4 + (j + 2)
                U[global_u][col] += coeff
            for v_idx, coeff in v_coeffs.items():
                j, k = divmod(v_idx, 2)
                global_v = (j + 2) * 6 + (k + k_offset)
                V[global_v][col] += coeff
            for w_idx, coeff in w_coeffs.items():
                k, i = divmod(w_idx, 2)
                global_w = (k + k_offset) * 2 + i
                W[global_w][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}
