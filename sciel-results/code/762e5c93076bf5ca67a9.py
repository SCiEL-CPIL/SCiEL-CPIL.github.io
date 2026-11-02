import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    U_strassen = np.array([
        [1, 0, 1, 1, 0, 1, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 1]
    ])
    V_strassen = np.array([
        [1, 0, 1, 0, -1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, -1],
        [0, 1, 0, 1, 0, -1, 0]
    ])
    W_strassen = np.array([
        [1, 0, 0, -1, 0, 0, -1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 1]
    ])
    
    # Schoolbook 2x2x1 decomposition
    # U: 4x4, V: 4x4, W: 4x4
    U_school = np.eye(4)
    V_school = np.eye(4)
    W_school = np.eye(4)
    
    # Build 2x2x15 using 7 Strassen blocks (cols 0-13) and 1 Schoolbook block (col 14)
    # Total rank: 7*7 + 4 = 53
    
    total_rank = 53
    U_full = np.zeros((4, total_rank))
    V_full = np.zeros((30, total_rank)) # m*p = 2*15 = 30
    W_full = np.zeros((30, total_rank)) # p*n = 15*2 = 30
    
    # Fill Strassen blocks
    for block_idx in range(7):
        rank_offset = block_idx * 7
        # V indices: block uses cols 2*block_idx and 2*block_idx+1
        # V row indices: j*p + k
        # j=0, k=0,1 -> rows 0, 1
        # j=1, k=0,1 -> rows 15, 16
        v_start = block_idx * 2
        # Map V_strassen (4x7) to V_full
        # V_strassen rows correspond to e,f,g,h (which are B[0,0], B[0,1], B[1,0], B[1,1])
        # In V_full, these are at rows:
        # B[0,0] -> 0*15 + 0 = 0
        # B[0,1] -> 0*15 + 1 = 1
        # B[1,0] -> 1*15 + 0 = 15
        # B[1,1] -> 1*15 + 1 = 16
        
        # We need to place V_strassen[0] into V_full[0, rank_offset:rank_offset+7]
        # V_strassen[1] into V_full[1, ...]
        # V_strassen[2] into V_full[15, ...]
        # V_strassen[3] into V_full[16, ...]
        
        V_full[0, rank_offset:rank_offset+7] = V_strassen[0]
        V_full[1, rank_offset:rank_offset+7] = V_strassen[1]
        V_full[15, rank_offset:rank_offset+7] = V_strassen[2]
        V_full[16, rank_offset:rank_offset+7] = V_strassen[3]
        
        # U is same for all blocks
        U_full[:, rank_offset:rank_offset+7] = U_strassen
        
        # W indices: block produces C[0,0], C[0,1], C[1,0], C[1,1]
        # W row indices: k*n + i
        # C[0,0] -> 0*2 + 0 = 0
        # C[0,1] -> 1*2 + 0 = 2
        # C[1,0] -> 0*2 + 1 = 1
        # C[1,1] -> 1*2 + 1 = 3
        
        W_full[0, rank_offset:rank_offset+7] = W_strassen[0]
        W_full[2, rank_offset:rank_offset+7] = W_strassen[1]
        W_full[1, rank_offset:rank_offset+7] = W_strassen[2]
        W_full[3, rank_offset:rank_offset+7] = W_strassen[3]
        
    # Fill Schoolbook block for col 14
    rank_offset = 49
    # V indices: col 14
    # B[0,14] -> 0*15 + 14 = 14
    # B[1,14] -> 1*15 + 14 = 29
    
    V_full[14, rank_offset:rank_offset+4] = V_school[0]
    V_full[15, rank_offset:rank_offset+4] = V_school[1] # Wait, V_school[1] is for B[0,1]?
    # Schoolbook for 2x2x1:
    # m1 = a * B[0,0]
    # m2 = b * B[1,0] ? No, b * B[0,1]?
    # C[0,0] = a B[0,0] + b B[1,0]
    # C[0,1] = a B[0,1] + b B[1,1]
    # For 2x2x1, B is 2x1.
    # C[0] = a B[0] + b B[1]
    # C[1] = c B[0] + d B[1]
    # m1 = a B[0]
    # m2 = b B[1]
    # m3 = c B[0]
    # m4 = d B[1]
    # U = I
    # V = I (rows for B[0], B[1]) -> But V is 2x1, so V is 2x4?
    # V[0] = [1, 0, 0, 0]
    # V[1] = [0, 1, 0, 0]
    # Wait, V shape is (m*p, rank) = (2*1, 4) = (2,4).
    # My V_full is (30, 53).
    # For the single column, we need V rows for B[0,14] and B[1,14].
    # V_full[14, 49:53] should be [1, 0, 0, 0]
    # V_full[29, 49:53] should be [0, 1, 0, 0]
    
    V_full[14, rank_offset:rank_offset+4] = [1, 0, 0, 0]
    V_full[29, rank_offset:rank_offset+4] = [0, 1, 0, 0]
    
    # U_full is I
    U_full[:, rank_offset:rank_offset+4] = np.eye(4)
    
    # W indices: C[0,14], C[1,14]
    # C[0,14] -> 14*2 + 0 = 28
    # C[1,14] -> 14*2 + 1 = 29
    # W[0] = [1, 0, 0, 0]
    # W[1] = [0, 1, 0, 0]
    # W[2] = [0, 0, 1, 0]
    # W[3] = [0, 0, 0, 1]
    
    W_full[28, rank_offset:rank_offset+4] = [1, 0, 0, 0]
    W_full[29, rank_offset:rank_offset+4] = [0, 1, 0, 0]
    # Wait, C[0,14] uses m1 and m2?
    # C[0,14] = a B[0,14] + b B[1,14] = m1 + m2
    # C[1,14] = c B[0,14] + d B[1,14] = m3 + m4
    # So W[0] (for C[0,14]) should be [1, 1, 0, 0]
    # W[1] (for C[1,14]) should be [0, 0, 1, 1]
    
    W_full[28, rank_offset:rank_offset+4] = [1, 1, 0, 0]
    W_full[29, rank_offset:rank_offset+4] = [0, 0, 1, 1]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
