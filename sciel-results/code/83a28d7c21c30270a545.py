import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    # Strassen patterns for 2x2x2: (U_coeffs, V_coeffs, W_coeffs)
    # Local indices: A: 0=a00, 1=a01, 2=a10, 3=a11
    #                 B: 0=b00, 1=b01, 2=b10, 3=b11
    #                 C: 0=c00, 1=c01, 2=c10, 3=c11
    strassen_patterns = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),           # M1
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),           # M2
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),         # M3
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),         # M4
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),           # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),               # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),               # M7
    ]
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # A is 2x4: split columns into 2 blocks of 2
    # B is 4x8: split rows into 2 blocks of 2, columns into 4 blocks of 2
    # C is 2x8: rows 0-1, columns in 4 blocks of 2
    
    for a_block in range(2):
        # A block: rows 0-1, cols 2*a_block .. 2*a_block+1
        for b_block in range(4):
            # B block: rows 2*a_block .. 2*a_block+1 (must match A cols), cols 2*b_block .. 2*b_block+1
            # C block: rows 0-1, cols 2*b_block .. 2*b_block+1
            
            a_col_start = 2 * a_block
            b_row_start = 2 * a_block
            c_col_start = 2 * b_block
            
            base_col = 7 * (4 * a_block + b_block)
            
            for s, (u_c, v_c, w_c) in enumerate(strassen_patterns):
                col = base_col + s
                
                # Fill U: A indices (i, j) where i in {0,1}, j in {a_col_start, a_col_start+1}
                for local_idx, val in u_c.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = li
                    global_j = a_col_start + lj
                    U[global_i * m + global_j, col] += val
                
                # Fill V: B indices (j, k) where j in {b_row_start, b_row_start+1}, k in {c_col_start, c_col_start+1}
                for local_idx, val in v_c.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    global_j = b_row_start + lj
                    global_k = c_col_start + lk
                    V[global_j * p + global_k, col] += val
                
                # Fill W: C indices (i, k) where i in {0,1}, k in {c_col_start, c_col_start+1}
                # W convention: W[k*n + i, r]
                for local_idx, val in w_c.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    global_i = li
                    global_k = c_col_start + lk
                    W[global_k * n + global_i, col] += val
    
    return {"U": U, "V": V, "W": W}
