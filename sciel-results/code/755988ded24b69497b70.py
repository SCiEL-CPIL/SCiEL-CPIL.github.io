import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The problem decomposes into two independent 2x2x8 multiplications.
    # Each 2x2x8 decomposes into four independent 2x2x2 multiplications.
    # Rank of 2x2x2 is 7 (Strassen).
    # Total rank = 2 * 4 * 7 = 56.
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Strassen 2x2x2 base blocks
    # U_local, V_local, W_local are 4x7 matrices
    # Indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2) for A,B,C
    # Note: The problem uses 0-based indexing in the arrays.
    # A[0,0]=1, A[0,1]=2, A[1,0]=3, A[1,1]=4 in 1-based?
    # Let's stick to the local 0,1,2,3 mapping used in the current code.
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
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
    
    # Fill the global matrices
    # j_group: 0 for A[:,0:2], 1 for A[:,2:4]
    # k_group: 0..3 for the 4 column pairs of B (and C)
    
    for j_group in range(2):
        # Map local A indices (0..3) to global A indices
        # A is 2x4. Index = i * 4 + k.
        # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global k = j_group * 2 + local_k
        # Global i = local_i
        a_map = [
            0 * 4 + (j_group * 2 + 0), # (0, j_group*2)
            0 * 4 + (j_group * 2 + 1), # (0, j_group*2+1)
            1 * 4 + (j_group * 2 + 0), # (1, j_group*2)
            1 * 4 + (j_group * 2 + 1)  # (1, j_group*2+1)
        ]
        
        for k_group in range(4):
            # Map local B indices (0..3) to global B indices
            # B is 4x8. Index = k * 8 + j.
            # Local B indices correspond to the 2x2 block of B.
            # The rows of this block are global k = j_group*2 + local_k_row?
            # Wait.
            # In the sub-problem 2x2x8:
            # A_sub is 2x2 (rows 0,1; cols 0,1).
            # B_sub is 2x8 (rows 0,1; cols 0..7).
            # C_sub is 2x8 (rows 0,1; cols 0..7).
            
            # Global A rows: 0,1. Global A cols: j_group*2, j_group*2+1.
            # Global B rows: j_group*2, j_group*2+1.
            # Global B cols: k_group*2, k_group*2+1.
            
            # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global B index = global_row * 8 + global_col
            b_map = [
                (j_group * 2 + 0) * 8 + (k_group * 2 + 0),
                (j_group * 2 + 0) * 8 + (k_group * 2 + 1),
                (j_group * 2 + 1) * 8 + (k_group * 2 + 0),
                (j_group * 2 + 1) * 8 + (k_group * 2 + 1)
            ]
            
            # Map local C indices (0..3) to global C indices
            # C is 2x8. Index = j * 2 + i (based on W[k*n+i] where k is col, i is row).
            # So index = global_col * 2 + global_row.
            # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global C col: k_group*2, k_group*2+1.
            # Global C row: 0,1.
            c_map = [
                (k_group * 2 + 0) * 2 + 0, # col 0, row 0
                (k_group * 2 + 1) * 2 + 0, # col 1, row 0
                (k_group * 2 + 0) * 2 + 1, # col 0, row 1
                (k_group * 2 + 1) * 2 + 1  # col 1, row 1
            ]
            
            base_r = (j_group * 4 + k_group) * 7
            
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
        # Should not happen, but safety check
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
