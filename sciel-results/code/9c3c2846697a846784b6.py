import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 100  # safety margin before 120s limit
    
    n, m, p = 3, 4, 8
    total_nm, total_mp, total_pn = n*m, m*p, p*n  # 12, 32, 24
    
    rank = 88
    U = np.zeros((total_nm, rank), dtype=float)
    V = np.zeros((total_mp, rank), dtype=float)
    W = np.zeros((total_pn, rank), dtype=float)
    
    # Part 1: i=0, schoolbook for 1x4x8 (rank 32)
    col = 0
    for j in range(m):
        for k in range(p):
            U[j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen on 2x2 blocks (rank 56)
    # Strassen 2x2x2 tensors
    s_U = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]  
    ])
    s_V = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    s_W = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, -1], 
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ])
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                col_t = col + t
                # Build column vectors for this Strassen tensor
                u_col = np.zeros(total_nm)
                v_col = np.zeros(total_mp)
                w_col = np.zeros(total_pn)
                
                for l in range(4):
                    val_u = s_U[t, l]
                    val_v = s_V[t, l]
                    val_w = s_W[t, l]
                    
                    if val_u != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li  # i index (1 or 2)
                        gj = j_start + lj
                        u_col[gi*m + gj] += val_u
                    
                    if val_v != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        v_col[gj*p + gk] += val_v
                    
                    if val_w != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        w_col[gk*n + gi] += val_w
                
                U[:, col_t] = u_col
                V[:, col_t] = v_col
                W[:, col_t] = w_col
            
            col += 7
    
    # Check if we're running out of time
    elapsed = time.time() - start
    if elapsed > deadline:
        # Return what we have (should be complete by now)
        pass
    
    return {"U": U, "V": V, "W": W}
