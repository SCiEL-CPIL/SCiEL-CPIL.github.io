import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # U1, V1, W1 shapes (4, 7)
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Term 1: (a00+a11)(b00+b11) -> c00, c11
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # Term 2: (a10+a11)b00 -> c10
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1 # c10: k=0, i=1 -> 0*2+1=1
    
    # Term 3: a00(b01-b11) -> c01
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[2, 2] = 1 # c01: k=1, i=0 -> 1*2+0=2
    
    # Term 4: a11(b10-b00) -> c10
    U1[3, 3] = 1
    V1[2, 3] = 1; V1[0, 3] = -1
    W1[1, 3] = 1 # c10
    
    # Term 5: (a00+a01)b11 -> -c00, c11
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[3, 4] = 1
    
    # Term 6: (a10-a00)(b00+b01) -> c11
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # Term 7: (a01-a11)(b10+b11) -> c00
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    # Schoolbook decomposition for 3x3
    # U2, V2, W2 shapes (9, 27)
    U2 = np.zeros((9, 27))
    V2 = np.zeros((9, 27))
    W2 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U2[i * 3 + j, col] = 1
                V2[j * 3 + k, col] = 1
                W2[k * 3 + i, col] = 1
                col += 1

    # Combine
    R1 = 7
    R2 = 27
    Rank = R1 * R2
    
    U = np.zeros((6 * 6, Rank))
    V = np.zeros((6 * 6, Rank))
    W = np.zeros((6 * 6, Rank))
    
    # Indices:
    # i = i1 * 3 + i2
    # j = j1 * 3 + j2
    # k = k1 * 3 + k2
    # Rank index r = r1 * 27 + r2
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            
            # Extract non-zero entries from U1, V1, W1 for r1
            # This is inefficient, but R1=7 is small.
            # Better: precompute sparse lists
            
    # Precompute sparse indices for U1, V1, W1
    U1_sparse = {}
    V1_sparse = {}
    W1_sparse = {}
    for r1 in range(R1):
        U1_sparse[r1] = [(idx, U1[idx, r1]) for idx in range(4) if U1[idx, r1] != 0]
        V1_sparse[r1] = [(idx, V1[idx, r1]) for idx in range(4) if V1[idx, r1] != 0]
        W1_sparse[r1] = [(idx, W1[idx, r1]) for idx in range(4) if W1[idx, r1] != 0]
        
    U2_sparse = {}
    V2_sparse = {}
    W2_sparse = {}
    for r2 in range(R2):
        U2_sparse[r2] = [(idx, U2[idx, r2]) for idx in range(9) if U2[idx, r2] != 0]
        V2_sparse[r2] = [(idx, V2[idx, r2]) for idx in range(9) if V2[idx, r2] != 0]
        W2_sparse[r2] = [(idx, W2[idx, r2]) for idx in range(9) if W2[idx, r2] != 0]

    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            
            for (i1_idx, u1_val) in U1_sparse[r1]:
                for (i2_idx, u2_val) in U2_sparse[r2]:
                    # i1_idx is 0..3 for 2x2. i1 = i1_idx // 2, i2_rem = i1_idx % 2?
                    # Wait, U1 index is i * 2 + j.
                    # So i1 = i1_idx // 2, j1 = i1_idx % 2.
                    # U2 index is i * 3 + j.
                    # So i2 = i2_idx // 3, j2 = i2_idx % 3.
                    
                    i1 = i1_idx // 2
                    j1 = i1_idx % 2
                    i2 = i2_idx // 3
                    j2 = i2_idx % 3
                    
                    i = i1 * 3 + i2
                    j = j1 * 3 + j2
                    U[i * 6 + j, r] += u1_val * u2_val
                    
            for (j1_idx, v1_val) in V1_sparse[r1]:
                for (j2_idx, v2_val) in V2_sparse[r2]:
                    j1 = j1_idx // 2
                    k1 = j1_idx % 2
                    j2 = j2_idx // 3
                    k2 = j2_idx % 3
                    
                    j = j1 * 3 + j2
                    k = k1 * 3 + k2
                    V[j * 6 + k, r] += v1_val * v2_val
                    
            for (k1_idx, w1_val) in W1_sparse[r1]:
                for (k2_idx, w2_val) in W2_sparse[r2]:
                    # W1 index is k * 2 + i.
                    k1 = k1_idx // 2
                    i1 = k1_idx % 2
                    k2 = k2_idx // 3
                    i2 = k2_idx % 3
                    
                    k = k1 * 3 + k2
                    i = i1 * 3 + i2
                    W[k * 6 + i, r] += w1_val * w2_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
