import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition using 6 blocks of 2x2 Strassen
    # Strassen rank-7 decomposition for 2x2x2
    # U (4x7), V (4x7), W (4x7) for local 2x2 blocks
    
    # Strassen terms for A(2x2) * B(2x2)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[C00, C01], [C10, C11]]
    
    # U factors for A indices (0:a, 1:b, 2:c, 3:d)
    U_strassen = np.array([
        [1, 0, 0, 1, 0, 0, 0],  # M1: a+d
        [0, 0, 0, 0, 1, 0, 0],  # M2: e (wait, M2 is (c+d)e, so U part is c+d)
        [0, 1, 0, 0, 0, 0, 0],  # M3: a
        [0, 0, 0, 1, 0, 0, 0],  # M4: d
        [1, 1, 0, 0, 0, 0, 0],  # M5: a+b
        [1, 0, -1, 0, 0, 0, 0], # M6: c-a (wait, standard is (c-a)(e+f))
        [0, 1, 0, -1, 0, 0, 0]  # M7: b-d
    ])
    
    # Correction: Standard Strassen
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
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
    # B indices: 0:e, 1:f, 2:g, 3:h
    V_s = np.zeros((4, 7))
    V_s[0, 0] = 1; V_s[3, 0] = 1  # M1
    V_s[0, 1] = 1                 # M2
    V_s[1, 2] = 1; V_s[3, 2] = -1 # M3
    V_s[2, 3] = 1; V_s[0, 3] = -1 # M4
    V_s[3, 4] = 1                 # M5
    V_s[0, 5] = 1; V_s[1, 5] = 1  # M6
    V_s[2, 6] = 1; V_s[3, 6] = 1  # M7
    
    # W (C part)
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11
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
        
        # U: same for all blocks (A is global 2x2)
        U_full[:, rank_offset:rank_offset+7] = U_s
        
        # V: B is 2x12. Block t uses columns 2t, 2t+1.
        # B indices in V_full:
        # Row 0..11: B[0, 0..11]
        # Row 12..23: B[1, 0..11]
        # Local B index 0 -> B[0, 2t] -> Row 2t
        # Local B index 1 -> B[0, 2t+1] -> Row 2t+1
        # Local B index 2 -> B[1, 2t] -> Row 12 + 2t
        # Local B index 3 -> B[1, 2t+1] -> Row 12 + 2t + 1
        
        # Map local V_s rows to global V_full rows
        row_map_v = [2*t, 2*t+1, 12+2*t, 12+2*t+1]
        for local_idx, global_idx in enumerate(row_map_v):
            V_full[global_idx, rank_offset:rank_offset+7] = V_s[local_idx, :]
            
        # W: C is 2x12. Block t uses columns 2t, 2t+1.
        # W index convention: W[k*n + i, r] where k is col, i is row?
        # Task says: W[k*n+i, r].
        # k=col, i=row.
        # C[0, 2t] -> k=2t, i=0 -> idx 2t*2 + 0 = 4t
        # C[0, 2t+1] -> k=2t+1, i=0 -> idx (2t+1)*2 + 0 = 4t+2
        # C[1, 2t] -> k=2t, i=1 -> idx 2t*2 + 1 = 4t+1
        # C[1, 2t+1] -> k=2t+1, i=1 -> idx (2t+1)*2 + 1 = 4t+3
        
        row_map_w = [4*t, 4*t+2, 4*t+1, 4*t+3]
        for local_idx, global_idx in enumerate(row_map_w):
            W_full[global_idx, rank_offset:rank_offset+7] = W_s[local_idx, :]

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
