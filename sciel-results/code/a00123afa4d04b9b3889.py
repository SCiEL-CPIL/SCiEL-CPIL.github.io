import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen 2x2x2 blocks
    
    # Local Strassen factors for 2x2x2
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u, v, w are lists of 7 vectors of length 4
    
    U_local = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    V_local = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [-1, 1, 0, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    W_local = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 0, 1, -1],# M2: C10-C11
        [0, 1, 0, 1], # M3: C01+C11
        [1, 0, 1, 0], # M4: C00+C10
        [-1, 1, 0, 0],# M5: -C00+C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 1]  # M7: C00+C11
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 4 blocks
    # Block 0: A_L (cols 0-1), B_LL (rows 0-1, cols 0-1)
    # Block 1: A_R (cols 2-3), B_RL (rows 2-3, cols 0-1)
    # Block 2: A_L (cols 0-1), B_LR (rows 0-1, cols 2-3)
    # Block 3: A_R (cols 2-3), B_RR (rows 2-3, cols 2-3)
    
    blocks = [
        (0, 0, 0), # (a_col_offset, b_row_offset, b_col_offset)
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2)
    ]
    
    col_idx = 0
    for a_off, b_r_off, b_c_off in blocks:
        for c in range(7):
            # Add U factors
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = U_local[c][local_idx]
                    if val != 0:
                        global_i = li
                        global_j = lj + a_off
                        u_idx = global_i * m + global_j
                        U[u_idx][col_idx] += val
                        
            # Add V factors
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    val = V_local[c][local_idx]
                    if val != 0:
                        global_j = lj + b_r_off
                        global_k = lk + b_c_off
                        v_idx = global_j * p + global_k
                        V[v_idx][col_idx] += val
                        
            # Add W factors
            for li in range(2):
                for lk in range(2):
                    local_idx = li * 2 + lk
                    val = W_local[c][local_idx]
                    if val != 0:
                        global_i = li
                        global_k = lk + b_c_off
                        # Trace convention: W[k*n + i, r]
                        w_idx = global_k * n + global_i
                        W[w_idx][col_idx] += val
                        
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
