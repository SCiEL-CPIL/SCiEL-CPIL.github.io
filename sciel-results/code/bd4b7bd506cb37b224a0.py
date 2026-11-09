import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    # We will construct 4 blocks of 2x3x2
    # Each block has rank 11 (7 from Strassen 2x2x2 + 4 from Schoolbook 2x1x2)
    
    # Strassen 2x2x2 factors
    # n=2, m=2, p=2
    # U (4, 7), V (4, 7), W (4, 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (a11-a10)(b00+b10) -> c00 +, c11 +
    # Indices: a11 is 1*2+1=3, a10 is 1*2+0=2. b00 is 0*2+0=0, b10 is 1*2+0=2.
    # c00 is 0*2+0=0, c11 is 1*2+1=3.
    U_s[3, 0] = 1; U_s[2, 0] = -1
    V_s[0, 0] = 1; V_s[2, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (a00+a10)b00 -> c10 +, c11 - (wait, c11 has -M2? No, c22 has -M2)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M6 - M5 - M7
    # c21 = M2 + M5 - M4 + M7
    # c22 = M1 - M2 + M3 + M6
    # M2 is in c21 (+) and c22 (-).
    # a00 is 0, a10 is 2. b00 is 0.
    # c21 is 1*2+0=2. c22 is 1*2+1=3.
    U_s[0, 1] = 1; U_s[2, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # M3 = (a01+a11)b11 -> c12 (+), c22 (+)
    # a01 is 1, a11 is 3. b11 is 3.
    # c12 is 0*2+1=1. c22 is 3.
    U_s[1, 2] = 1; U_s[3, 2] = 1
    V_s[3, 2] = 1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # M4 = a00(b11-b01) -> c00 (+), c21 (-)
    # a00 is 0. b11 is 3, b01 is 1.
    # c00 is 0. c21 is 2.
    U_s[0, 3] = 1
    V_s[3, 3] = 1; V_s[1, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = -1
    
    # M5 = a11(b00-b10) -> c00 (-), c12 (-), c21 (+)
    # a11 is 3. b00 is 0, b10 is 2.
    # c00 is 0. c12 is 1. c21 is 2.
    U_s[3, 4] = 1
    V_s[0, 4] = 1; V_s[2, 4] = -1
    W_s[0, 4] = -1; W_s[1, 4] = -1; W_s[2, 4] = 1
    
    # M6 = (a00+a01)(b11-b10) -> c12 (+), c22 (+)
    # a00 is 0, a01 is 1. b11 is 3, b10 is 2.
    # c12 is 1. c22 is 3.
    U_s[0, 5] = 1; U_s[1, 5] = 1
    V_s[3, 5] = 1; V_s[2, 5] = -1
    W_s[1, 5] = 1; W_s[3, 5] = 1
    
    # M7 = (a10-a00)(b00+b01) -> c00 (+), c12 (-), c21 (+)
    # a10 is 2, a00 is 0. b00 is 0, b01 is 1.
    # c00 is 0. c12 is 1. c21 is 2.
    U_s[2, 6] = 1; U_s[0, 6] = -1
    V_s[0, 6] = 1; V_s[1, 6] = 1
    W_s[0, 6] = 1; W_s[1, 6] = -1; W_s[2, 6] = 1

    # Schoolbook 2x1x2 factors (Rank 4)
    # n=2, m=1, p=2
    # U (2, 4), V (2, 4), W (4, 4)
    U_sb = np.zeros((2, 4))
    V_sb = np.zeros((2, 4))
    W_sb = np.zeros((4, 4))
    
    # c00 = a00 b00
    U_sb[0, 0] = 1
    V_sb[0, 0] = 1
    W_sb[0, 0] = 1 # k=0, i=0
    
    # c01 = a00 b01
    U_sb[0, 1] = 1
    V_sb[1, 1] = 1
    W_sb[1, 1] = 1 # k=1, i=0
    
    # c10 = a10 b00
    U_sb[1, 2] = 1
    V_sb[0, 2] = 1
    W_sb[2, 2] = 1 # k=0, i=1
    
    # c11 = a10 b01
    U_sb[1, 3] = 1
    V_sb[1, 3] = 1
    W_sb[3, 3] = 1 # k=1, i=1

    # Combine into 2x3x2 (Rank 11)
    # n=2, m=3, p=2
    # U (6, 11), V (6, 11), W (4, 11)
    U_b = np.zeros((6, 11))
    V_b = np.zeros((6, 11))
    W_b = np.zeros((4, 11))
    
    # Strassen part uses A[0:2, 0:2] and B[0:2, :]
    # U indices: 0..3 correspond to A[0,0]..A[1,1]
    # V indices: 0..3 correspond to B[0,0]..B[1,1]
    # W indices: 0..3 correspond to C[0,0]..C[1,1]
    U_b[0:4, 0:7] = U_s
    V_b[0:4, 0:7] = V_s
    W_b[0:4, 0:7] = W_s
    
    # Schoolbook part uses A[:, 2] and B[2, :]
    # A[:, 2] is A[0,2] (idx 2) and A[1,2] (idx 5)
    # B[2, :] is B[2,0] (idx 4) and B[2,1] (idx 5)
    # U_sb indices 0,1 map to A[0,2], A[1,2] -> U_b[2], U_b[5]
    # V_sb indices 0,1 map to B[2,0], B[2,1] -> V_b[4], V_b[5]
    # W_sb indices 0..3 map to C[0,0]..C[1,1] -> W_b[0]..W_b[3]
    
    U_b[2, 7:11] = U_sb[0, :]
    U_b[5, 7:11] = U_sb[1, :]
    V_b[4, 7:11] = V_sb[0, :]
    V_b[5, 7:11] = V_sb[1, :]
    W_b[0:4, 7:11] = W_sb

    # Now tile 4 times for 2x3x8
    # n=2, m=3, p=8
    # U (6, 44), V (24, 44), W (16, 44)
    rank_total = 44
    U_final = np.zeros((6, rank_total))
    V_final = np.zeros((24, rank_total))
    W_final = np.zeros((16, rank_total))
    
    for t in range(4):
        col_offset = t * 11
        # U is same for all blocks
        U_final[:, col_offset:col_offset+11] = U_b
        
        # V: B is 3x8. Block t uses columns 2t, 2t+1.
        # B indices: j*8 + k.
        # For block t, j=0,1 use B[0, 2t], B[1, 2t] etc.
        # V_b indices 0..5 correspond to B[0,0]..B[2,1] in local 3x2.
        # Global indices:
        # Local B[0,0] -> Global B[0, 2t] -> idx 0*8 + 2t = 2t
        # Local B[0,1] -> Global B[0, 2t+1] -> idx 2t+1
        # Local B[1,0] -> Global B[1, 2t] -> idx 8 + 2t
        # Local B[1,1] -> Global B[1, 2t+1] -> idx 9 + 2t
        # Local B[2,0] -> Global B[2, 2t] -> idx 16 + 2t
        # Local B[2,1] -> Global B[2, 2t+1] -> idx 17 + 2t
        
        v_indices = [2*t, 2*t+1, 8+2*t, 9+2*t, 16+2*t, 17+2*t]
        for local_idx, global_idx in enumerate(v_indices):
            V_final[global_idx, col_offset:col_offset+11] = V_b[local_idx, :]
            
        # W: C is 2x8. Block t uses columns 2t, 2t+1.
        # W indices: k*2 + i.
        # Local C[0,0] -> Global C[0, 2t] -> k=2t, i=0 -> idx 2t*2 + 0 = 4t
        # Local C[0,1] -> Global C[0, 2t+1] -> k=2t+1, i=0 -> idx (2t+1)*2 + 0 = 4t+2
        # Local C[1,0] -> Global C[1, 2t] -> k=2t, i=1 -> idx 4t+1
        # Local C[1,1] -> Global C[1, 2t+1] -> k=2t+1, i=1 -> idx 4t+3
        
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        for local_idx, global_idx in enumerate(w_indices):
            W_final[global_idx, col_offset:col_offset+11] = W_b[local_idx, :]

    if time.time() > deadline:
        return None # Should not happen
        
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
