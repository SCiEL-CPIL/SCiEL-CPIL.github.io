import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s guard
    
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition using 6 blocks of 2x2 Strassen
    # Strassen rank-7 decomposition for 2x2x2
    
    # U (A part)
    U_s = np.zeros((4, 7))
    U_s[0, 0] = 1; U_s[3, 0] = 1  # M1
    U_s[2, 1] = 1; U_s[3, 1] = 1  # M2
    U_s[0, 2] = 1                 # M3
    U_s[3, 3] = 1                 # M4
    U_s[0, 4] = 1; U_s[1, 4] = 1  # M5
    U_s[2, 5] = 1; U_s[0, 5] = -1 # M6
    U_s[1, 6] = 1; U_s[3, 6] = -1 # M7
    
    # V (B part)
    V_s = np.zeros((4, 7))
    V_s[0, 0] = 1; V_s[3, 0] = 1  # M1
    V_s[0, 1] = 1                 # M2
    V_s[1, 2] = 1; V_s[3, 2] = -1 # M3
    V_s[2, 3] = 1; V_s[0, 3] = -1 # M4
    V_s[3, 4] = 1                 # M5
    V_s[0, 5] = 1; V_s[1, 5] = 1  # M6
    V_s[2, 6] = 1; V_s[3, 6] = 1  # M7
    
    # W (C part)
    W_s = np.zeros((4, 7))
    # C00 = M1 + M4 - M5 + M7
    W_s[0, 0] = 1; W_s[0, 3] = 1; W_s[0, 4] = -1; W_s[0, 6] = 1
    # C01 = M3 + M5
    W_s[1, 2] = 1; W_s[1, 4] = 1
    # C10 = M2 + M4
    W_s[2, 1] = 1; W_s[2, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W_s[3, 0] = 1; W_s[3, 1] = -1; W_s[3, 2] = 1; W_s[3, 5] = 1

    num_blocks = p // 2  # 6 blocks
    total_rank = num_blocks * 7
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for t in range(num_blocks):
        rank_offset = t * 7
        
        # U: same for all blocks
        U_full[:, rank_offset:rank_offset+7] = U_s
        
        # V: B is 2x12. Block t uses columns 2t, 2t+1.
        row_map_v = [2*t, 2*t+1, 12+2*t, 12+2*t+1]
        for local_idx, global_idx in enumerate(row_map_v):
            V_full[global_idx, rank_offset:rank_offset+7] = V_s[local_idx, :]
            
        # W: C is 2x12. Block t uses columns 2t, 2t+1.
        # W[k*n+i, r] with n=2.
        row_map_w = [4*t, 4*t+2, 4*t+1, 4*t+3]
        for local_idx, global_idx in enumerate(row_map_w):
            W_full[global_idx, rank_offset:rank_offset+7] = W_s[local_idx, :]

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
