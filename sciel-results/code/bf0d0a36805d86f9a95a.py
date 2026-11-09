import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Construct 2x2x2 schoolbook (rank 8)
    U2 = np.eye(4).reshape(2, 2, 2).reshape(2, 4) # No, schoolbook is simpler
    # U2[i, r] where r = i*2 + j? No, r = i*2 + j is for U[i, j]?
    # Schoolbook: r = i*m + j.
    # For 2x2x2: r = i*2 + j.
    # U[i, r] = 1 if r = i*2 + j.
    # V[j, r] = 1 if r = i*2 + j.
    # W[k, r] = 1 if r = i*2 + j? No, W depends on k.
    # T[i,j,k] = 1.
    # Sum_r U[i,r] V[j,r] W[k,r] = 1.
    # If r = i*2 + j, then U[i,r]=1, V[j,r]=1.
    # We need W[k,r] to be 1 for all k? No.
    # Schoolbook for 2x2x2:
    # r = i*2 + j.
    # U[i, r] = 1.
    # V[j, r] = 1.
    # W[k, r] = 1?
    # Then Sum_k W[k,r] = 2.
    # We need Sum_r ... = 1.
    # So W[k, r] = 1/2?
    # 1/2 is half-integer. Allowed.
    
    U2 = np.zeros((2, 8))
    V2 = np.zeros((2, 8))
    W2 = np.zeros((2, 8))
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U2[i, r] = 1
            V2[j, r] = 1
            W2[0, r] = 0.5
            W2[1, r] = 0.5
            
    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i, :] * V2[j, :] * W2[k, :])
                if abs(val - 1.0) > 1e-8:
                    print("2x2x2 fail")
                    
    # 2. Construct 4x4x4 from 2x2x2 (rank 64)
    U4 = np.zeros((4, 64))
    V4 = np.zeros((4, 64))
    W4 = np.zeros((4, 64))
    for r1 in range(8):
        for r2 in range(8):
            r = r1 * 8 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U4[i, r] += U2[i1, r1] * U2[i2, r2]
                    V4[i, r] += V2[i1, r1] * V2[i2, r2] # Wait, V index is j.
                    # V4[j, r] where j = j1*2 + j2.
                    # So we need to loop j1, j2.
                    
    # Redo 4x4x4 construction properly
    U4 = np.zeros((4, 64))
    V4 = np.zeros((4, 64))
    W4 = np.zeros((4, 64))
    for r1 in range(8):
        for r2 in range(8):
            r = r1 * 8 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U4[i, r] = U2[i1, r1] * U2[i2, r2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    V4[j, r] = V2[j1, r1] * V2[j2, r2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    W4[k, r] = W2[k1, r1] * W2[k2, r2]
                    
    # 3. Construct 4x4x1 schoolbook (rank 16)
    U16 = np.zeros((4, 16))
    V16 = np.zeros((4, 16))
    W16 = np.zeros((1, 16)) # k is fixed to 4 (index 4 in 0..4)
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U16[i, r] = 1
            V16[j, r] = 1
            W16[0, r] = 1
            
    # 4. Construct 4x1x5 rank 1
    U1 = np.zeros((4, 1))
    V1 = np.zeros((1, 1))
    W1 = np.zeros((5, 1))
    for i in range(4):
        U1[i, 0] = 1
    V1[0, 0] = 1
    for k in range(5):
        W1[k, 0] = 1
        
    # Combine
    rank_total = 64 + 16 + 1
    U = np.zeros((20, rank_total))
    V = np.zeros((25, rank_total))
    W = np.zeros((20, rank_total))
    
    # Part 1: 4x4x4 (i 0..3, j 0..3, k 0..3)
    U[0:4, 0:64] = U4
    V[0:4, 0:64] = V4
    W[0:4, 0:64] = W4
    
    # Part 2: 4x4x1 (i 0..3, j 0..3, k 4)
    U[0:4, 64:80] = U16
    V[0:4, 64:80] = V16
    W[4, 64:80] = W16[0]
    
    # Part 3: 4x1x5 (i 0..3, j 4, k 0..4)
    U[0:4, 80:81] = U1
    V[4, 80:81] = V1
    W[0:5, 80:81] = W1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
