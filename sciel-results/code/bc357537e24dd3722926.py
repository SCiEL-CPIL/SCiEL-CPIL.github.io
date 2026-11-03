import numpy as np

def construct():
    n, m, p = 6, 6, 6
    # Block size is 3
    bs = 3
    # Number of blocks per dimension is 2
    nb = n // bs
    
    # Strassen coefficients for 2x2 block multiplication
    # u_r[bc], v_r[b'c'], w_r[b''c'']
    # bc index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
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
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_coeffs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11(-)
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00(-), C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    rank = 7 * bs * bs * bs
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for r in range(7):
        u_c = U_coeffs[r]
        v_c = V_coeffs[r]
        w_c = W_coeffs[r]
        
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # U vector
                    for b in range(nb):
                        for c in range(nb):
                            idx_bc = b * 2 + c
                            coeff = u_c[idx_bc]
                            if coeff != 0:
                                row_A = b * bs + i
                                col_A = c * bs + k
                                flat_A = row_A * m + col_A
                                U[flat_A, col] = coeff
                    
                    # V vector
                    for b in range(nb):
                        for c in range(nb):
                            idx_bc = b * 2 + c
                            coeff = v_c[idx_bc]
                            if coeff != 0:
                                row_B = b * bs + k
                                col_B = c * bs + j
                                flat_B = row_B * p + col_B
                                V[flat_B, col] = coeff
                                
                    # W vector
                    for b in range(nb):
                        for c in range(nb):
                            idx_bc = b * 2 + c
                            coeff = w_c[idx_bc]
                            if coeff != 0:
                                row_C = b * bs + i
                                col_C = c * bs + j
                                flat_C = row_C * n + col_C
                                W[flat_C, col] = coeff
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
