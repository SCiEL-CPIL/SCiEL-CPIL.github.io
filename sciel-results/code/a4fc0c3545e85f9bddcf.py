import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Local indices:
    # U (A): 0=A00, 1=A01, 2=A10, 3=A11
    # V (B): 0=B00, 1=B01, 2=B10, 3=B11
    # W (C): 0=C00, 1=C01, 2=C10, 3=C11
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)*(B00+B11) -> C00, C11
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1
    
    # M2: (A10+A11)*B00 -> C10, C11
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1
    
    # M3: A00*(B01-B11) -> C01, C11
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1
    
    # M4: A11*(B10-B00) -> C00, C10
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1
    
    # M5: (A00+A01)*B11 -> C00, C01
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1
    
    # M6: (A10-A00)*(B00+B01) -> C11
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1
    
    # M7: (A01-A11)*(B10+B11) -> C00
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # We have 4 column pairs in C (j=0..3)
    # For each j, we have two 2x2x2 multiplications:
    # 1. A[:, 0:2] * B[0:2, 2j:2j+2]
    # 2. A[:, 2:4] * B[2:4, 2j:2j+2]
    
    for j in range(4):
        # s=0: A_left, B_top
        # s=1: A_right, B_bottom
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Map local A indices (0..3) to global A indices
            # Local: 00, 01, 10, 11
            if s == 0:
                # A[:, 0:2] -> cols 0,1
                a_map = [0, 1, 4, 5]
            else:
                # A[:, 2:4] -> cols 2,3
                a_map = [2, 3, 6, 7]
                
            # Map local B indices (0..3) to global B indices
            # Local: 00, 01, 10, 11 (rows 0,1 of the 2x2 block)
            # Global B is 4x8.
            # Block is rows (2*s .. 2*s+1) and cols (2*j .. 2*j+1)
            b_row_start = 2 * s
            b_col_start = 2 * j
            b_map = [
                b_row_start * 8 + b_col_start,
                b_row_start * 8 + b_col_start + 1,
                (b_row_start + 1) * 8 + b_col_start,
                (b_row_start + 1) * 8 + b_col_start + 1
            ]
            
            # Map local C indices (0..3) to global C indices
            # Global C is 2x8 (n=2, p=8).
            # Convention: W[k*n + i, r] where k is col, i is row.
            # Local C: 00, 01, 10, 11
            # Local 00 -> global row 0, col 2*j
            # Local 01 -> global row 0, col 2*j+1
            # Local 10 -> global row 1, col 2*j
            # Local 11 -> global row 1, col 2*j+1
            c_row = 0 # Local row 0
            c_col_start = 2 * j
            c_map = [
                c_col_start * 2 + 0,       # C[0, 2j] -> k=2j, i=0
                (c_col_start + 1) * 2 + 0, # C[0, 2j+1] -> k=2j+1, i=0
                c_col_start * 2 + 1,       # C[1, 2j] -> k=2j, i=1
                (c_col_start + 1) * 2 + 1  # C[1, 2j+1] -> k=2j+1, i=1
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
