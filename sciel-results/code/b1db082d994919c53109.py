import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Construct 3x3 Schoolbook Decomposition (Rank 27)
    # U3: (9, 27), V3: (9, 27), W3: (9, 27)
    rank3 = 27
    U3 = np.zeros((9, rank3))
    V3 = np.zeros((9, rank3))
    W3 = np.zeros((9, rank3))
    
    col3 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # U3 index: i*3 + j
                # V3 index: j*3 + k
                # W3 index: k*3 + i
                U3[i * 3 + j, col3] = 1
                V3[j * 3 + k, col3] = 1
                W3[k * 3 + i, col3] = 1
                col3 += 1

    # 2. Construct 2x2 Strassen Decomposition (Rank 7)
    # U2: (4, 7), V2: (4, 7), W2: (4, 7)
    rank2 = 7
    U2 = np.zeros((4, rank2))
    V2 = np.zeros((4, rank2))
    W2 = np.zeros((4, rank2))
    
    # Indices for 2x2 blocks:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00 (k=0,i=0), 1:10 (k=0,i=1), 2:01 (k=1,i=0), 3:11 (k=1,i=1)
    # Note: W2 index is k*2 + i.
    # C00 -> 0*2+0 = 0
    # C10 -> 0*2+1 = 1
    # C01 -> 1*2+0 = 2
    # C11 -> 1*2+1 = 3
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 3. Combine to 6x6x6
    total_rank = rank2 * rank3
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    for r2 in range(rank2):
        for r3 in range(rank3):
            r = r2 * rank3 + r3
            
            # Get non-zero indices and values from U2, V2, W2
            # And U3, V3, W3
            
            # Iterate over blocks
            for a_block in range(4):
                u2_val = U2[a_block, r2]
                if u2_val == 0: continue
                
                # Get inner indices for this a_block (I, J)
                I = a_block // 2
                J = a_block % 2
                
                # For each inner index in U3
                for a_inner in range(9):
                    u3_val = U3[a_inner, r3]
                    if u3_val == 0: continue
                    
                    # Global index for A
                    # a_block determines I, J. a_inner determines i, j.
                    # a_inner = i*3 + j
                    i = a_inner // 3
                    j = a_inner % 3
                    
                    # Global A index: (I*3 + i)*6 + (J*3 + j)
                    a_idx = (I * 3 + i) * 6 + (J * 3 + j)
                    
                    U[a_idx, r] = u2_val * u3_val
            
            for b_block in range(4):
                v2_val = V2[b_block, r2]
                if v2_val == 0: continue
                
                J = b_block // 2
                K = b_block % 2
                
                for b_inner in range(9):
                    v3_val = V3[b_inner, r3]
                    if v3_val == 0: continue
                    
                    j = b_inner // 3
                    k = b_inner % 3
                    
                    # Global B index: (J*3 + j)*6 + (K*3 + k)
                    b_idx = (J * 3 + j) * 6 + (K * 3 + k)
                    
                    V[b_idx, r] = v2_val * v3_val
            
            for c_block in range(4):
                w2_val = W2[c_block, r2]
                if w2_val == 0: continue
                
                K = c_block // 2
                I = c_block % 2
                
                for c_inner in range(9):
                    w3_val = W3[c_inner, r3]
                    if w3_val == 0: continue
                    
                    k = c_inner // 3
                    i = c_inner % 3
                    
                    # Global C index: (K*3 + k)*6 + (I*3 + i)
                    c_idx = (K * 3 + k) * 6 + (I * 3 + i)
                    
                    W[c_idx, r] = w2_val * w3_val

    # Verification (optional, but good for correctness during development)
    # We can skip this in production if time is tight, but let's ensure it's fast.
    # The construction is exact integer arithmetic, so it should be valid.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
