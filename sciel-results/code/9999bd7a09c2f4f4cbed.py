import numpy as np

def construct():
    n, m, p = 4, 4, 9
    # Total rank
    # 4 blocks of 2x2x9
    # Each 2x2x9 uses 4 Strassen (2x2x2) + 1 Standard (2x2x1)
    # Rank = 4 * (4 * 7 + 4) = 4 * 32 = 128
    
    R = 128
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Helper to add a decomposition
    # local_u, local_v, local_w are arrays of shape (4, r_sub) for U, V, W respectively?
    # Actually, for 2x2x2, U is 4x7, V is 4x7, W is 4x7.
    # For 2x2x1, U is 4x4, V is 2x4 (since p=1 locally? No, V size is m*p_local), W is 2x4.
    # We need to map local indices to global.
    
    # Define Strassen 2x2x2 matrices
    # U_s: 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float).T # Wait, U_s[i,j,r].
    # My previous derivation:
    # U_1 = [1, 0, 0, 1] -> col 0
    # U_2 = [0, 0, 1, 1] -> col 1
    # U_3 = [1, 0, 0, 0] -> col 2
    # U_4 = [0, 0, 0, 1] -> col 3
    # U_5 = [1, 1, 0, 0] -> col 4
    # U_6 = [-1, 0, 1, 0] -> col 5
    # U_7 = [0, 1, 0, -1] -> col 6
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    # Check U_s[0] = [1,0,1,0,1,-1,0]. Matches U_1..U_7 for index 0?
    # U_1[0]=1, U_2[0]=0, U_3[0]=1, U_4[0]=0, U_5[0]=1, U_6[0]=-1, U_7[0]=0. Yes.
    
    # V_s: 4x7
    # V_1 = [1, 0, 0, 1]
    # V_2 = [1, 0, 0, 0]
    # V_3 = [0, 1, 0, -1]
    # V_4 = [0, 0, 1, -1]
    # V_5 = [0, 0, 0, 1]
    # V_6 = [1, 1, 0, 0]
    # V_7 = [0, 0, 1, 1]
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=float)
    
    # W_s: 4x7
    # W_1 = [1, 0, 0, 1]
    # W_2 = [0, 0, 1, 1]
    # W_3 = [0, 1, 0, 1]
    # W_4 = [1, 0, 1, 0]
    # W_5 = [-1, 1, 0, 0]
    # W_6 = [0, 0, 0, 1]
    # W_7 = [1, 0, 0, 0]
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Standard 2x2x1 matrices
    # U_std: 4x4
    # r=0: i=0,j=0 -> [1,0,0,0]
    # r=1: i=0,j=1 -> [0,1,0,0]
    # r=2: i=1,j=0 -> [0,0,1,0]
    # r=3: i=1,j=1 -> [0,0,0,1]
    U_std = np.eye(4, dtype=float)
    
    # V_std: 2x4 (m=2, p_local=1)
    # r=0: j=0 -> [1, 0]
    # r=1: j=1 -> [0, 1]
    # r=2: j=0 -> [1, 0]
    # r=3: j=1 -> [0, 1]
    V_std = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    
    # W_std: 2x4 (p_local=1, n=2)
    # r=0: i=0 -> [1, 0]
    # r=1: i=0 -> [1, 0]
    # r=2: i=1 -> [0, 1]
    # r=3: i=1 -> [0, 1]
    W_std = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    # Mapping functions
    # Global A index: i_glob * 4 + j_glob
    # Global B index: j_glob * 9 + k_glob
    # Global W index: k_glob * 4 + i_glob
    
    r_offset = 0
    
    # Loop over 4 blocks of A
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # Corresponding B blocks: Top (rows 0,1) or Bot (rows 2,3)
    # C blocks: Top (rows 0,1) or Bot (rows 2,3)
    
    blocks = [
        # (i_start, j_start, j_b_start, i_c_start)
        (0, 0, 0, 0), # A_00 * B_top -> C_top
        (0, 2, 2, 0), # A_01 * B_bot -> C_top
        (2, 0, 0, 2), # A_10 * B_top -> C_bot
        (2, 2, 2, 2), # A_11 * B_bot -> C_bot
    ]
    
    for i_start, j_start, j_b_start, i_c_start in blocks:
        # For each of the 5 k-segments in B
        # k segments: (0,1), (2,3), (4,5), (6,7), (8,8)
        k_segments = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 8)]
        
        for k_start, k_end in k_segments:
            if k_end - k_start == 1:
                # Strassen 2x2x2
                # Local U: 4x7
                # Local V: 4x7
                # Local W: 4x7
                
                # Map U
                for li in range(2):
                    for lj in range(2):
                        gi = i_start + li
                        gj = j_start + lj
                        g_idx_u = gi * 4 + gj
                        for r in range(7):
                            U[g_idx_u, r_offset + r] += U_s[li * 2 + lj, r]
                
                # Map V
                for lj in range(2):
                    for lk in range(2):
                        gj = j_b_start + lj
                        gk = k_start + lk
                        g_idx_v = gj * 9 + gk
                        for r in range(7):
                            V[g_idx_v, r_offset + r] += V_s[lj * 2 + lk, r]
                
                # Map W
                for lk in range(2):
                    for li in range(2):
                        gk = k_start + lk
                        gi = i_c_start + li
                        g_idx_w = gk * 4 + gi
                        for r in range(7):
                            W[g_idx_w, r_offset + r] += W_s[lk * 2 + li, r]
                
                r_offset += 7
            else:
                # Standard 2x2x1
                # Local U: 4x4
                # Local V: 2x4
                # Local W: 2x4
                
                # Map U
                for li in range(2):
                    for lj in range(2):
                        gi = i_start + li
                        gj = j_start + lj
                        g_idx_u = gi * 4 + gj
                        for r in range(4):
                            U[g_idx_u, r_offset + r] += U_std[li * 2 + lj, r]
                
                # Map V
                for lj in range(2):
                    gj = j_b_start + lj
                    gk = k_start
                    g_idx_v = gj * 9 + gk
                    for r in range(4):
                        V[g_idx_v, r_offset + r] += V_std[lj, r]
                
                # Map W
                for li in range(2):
                    gk = k_start
                    gi = i_c_start + li
                    g_idx_w = gk * 4 + gi
                    for r in range(4):
                        W[g_idx_w, r_offset + r] += W_std[li, r]
                
                r_offset += 4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
