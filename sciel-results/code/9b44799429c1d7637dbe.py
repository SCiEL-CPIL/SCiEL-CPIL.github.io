import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize global matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Local Strassen factors for 2x2x2
    # U_loc[i*2+j, r]
    U_loc = np.zeros((4, 7), dtype=int)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1  # r=0: A00+A11
    U_loc[2, 1] = 1; U_loc[3, 1] = 1  # r=1: A10+A11
    U_loc[0, 2] = 1                   # r=2: A00
    U_loc[3, 3] = 1                   # r=3: A11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1  # r=4: A00+A01
    U_loc[2, 5] = 1; U_loc[0, 5] = -1 # r=5: A10-A00
    U_loc[1, 6] = 1; U_loc[3, 6] = -1 # r=6: A01-A11
    
    # V_loc[j*2+k, r]
    V_loc = np.zeros((4, 7), dtype=int)
    V_loc[0, 0] = 1; V_loc[3, 0] = 1  # r=0: B00+B11
    V_loc[0, 1] = 1                   # r=1: B00
    V_loc[1, 2] = 1; V_loc[3, 2] = -1 # r=2: B01-B11
    V_loc[2, 3] = 1; V_loc[0, 3] = -1 # r=3: B10-B00
    V_loc[3, 4] = 1                   # r=4: B11
    V_loc[0, 5] = 1; V_loc[1, 5] = 1  # r=5: B00+B01
    V_loc[2, 6] = 1; V_loc[3, 6] = 1  # r=6: B10+B11
    
    # W_loc[k*2+i, r]
    W_loc = np.zeros((4, 7), dtype=int)
    W_loc[0, 0] = 1
    W_loc[1, 1] = 1
    W_loc[2, 2] = 1
    W_loc[0, 3] = 1
    W_loc[1, 3] = 1
    W_loc[0, 4] = -1
    W_loc[2, 4] = 1
    W_loc[3, 5] = 1
    W_loc[0, 6] = 1
    W_loc[3, 0] = 1
    W_loc[3, 1] = -1
    W_loc[3, 2] = 1
    
    # Note: W_loc indices are k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Check W_loc against derived coefficients:
    # r=0 (M1): C00(+1), C11(+1) -> W[0,0]=1, W[3,0]=1. Correct.
    # r=1 (M2): C10(+1), C11(-1) -> W[2,1]=1, W[3,1]=-1. Correct.
    # r=2 (M3): C01(+1), C11(+1) -> W[1,2]=1, W[3,2]=1. Correct.
    # r=3 (M4): C00(+1), C10(+1) -> W[0,3]=1, W[2,3]=1. Correct.
    # r=4 (M5): C00(-1), C01(+1) -> W[0,4]=-1, W[1,4]=1. Correct.
    # r=5 (M6): C11(+1) -> W[3,5]=1. Correct.
    # r=6 (M7): C00(+1) -> W[0,6]=1. Correct.

    # Map 6 blocks
    for t in range(6):
        col_base = t * 7
        
        if t < 3:
            # A_left (j=0,1), B_top (k=2t, 2t+1)
            j_offset = 0
            k_offset = 2 * t
        else:
            # A_right (j=2,3), B_bot (k=2(t-3), 2(t-3)+1)
            j_offset = 2
            k_offset = 2 * (t - 3)
            
        # Copy U_loc
        # U global index: i*4 + j
        # U_loc index: i*2 + j_loc
        for i in range(2):
            for j_loc in range(2):
                j_glob = j_offset + j_loc
                u_idx_glob = i * 4 + j_glob
                u_idx_loc = i * 2 + j_loc
                U[u_idx_glob, col_base:col_base+7] += U_loc[u_idx_loc, :]
                
        # Copy V_loc
        # V global index: j*6 + k
        # V_loc index: j_loc*2 + k_loc
        for j_loc in range(2):
            j_glob = j_offset + j_loc
            for k_loc in range(2):
                k_glob = k_offset + k_loc
                v_idx_glob = j_glob * 6 + k_glob
                v_idx_loc = j_loc * 2 + k_loc
                V[v_idx_glob, col_base:col_base+7] += V_loc[v_idx_loc, :]
                
        # Copy W_loc
        # W global index: k*2 + i
        # W_loc index: k_loc*2 + i
        for k_loc in range(2):
            k_glob = k_offset + k_loc
            for i in range(2):
                w_idx_glob = k_glob * 2 + i
                w_idx_loc = k_loc * 2 + i
                W[w_idx_glob, col_base:col_base+7] += W_loc[w_idx_loc, :]
                
    return {"U": U, "V": V, "W": W}
