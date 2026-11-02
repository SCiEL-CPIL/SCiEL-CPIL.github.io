"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # We use a block decomposition:
    # Partition 6x6 into 3x3 blocks of 2x2 matrices.
    # Use schoolbook for the 3x3 block structure (rank 27).
    # Use Strassen for the 2x2 internal matrices (rank 7).
    # Total rank = 27 * 7 = 189.
    
    rank_3 = 27
    rank_2 = 7
    rank_total = rank_3 * rank_2
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # 1. Construct U3, V3, W3 for M3 schoolbook
    U3 = np.zeros((9, rank_3))
    V3 = np.zeros((9, rank_3))
    W3 = np.zeros((9, rank_3))
    
    r3 = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                # U3 indexed by (i1, j1)
                U3[i1 * 3 + j1, r3] = 1.0
                # V3 indexed by (j1, k1)
                V3[j1 * 3 + k1, r3] = 1.0
                # W3 indexed by (k1, i1)
                W3[k1 * 3 + i1, r3] = 1.0
                r3 += 1
                
    # 2. Construct U2, V2, W2 for M2 Strassen
    U2 = np.zeros((4, rank_2))
    V2 = np.zeros((4, rank_2))
    W2 = np.zeros((4, rank_2))
    
    # Strassen terms
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1 # C11 += M1
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C21 += M2
    W2[3, 1] = -1 # C22 -= M2
    
    # M3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C12 += M3
    W2[3, 2] = 1 # C22 += M3
    
    # M4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C11 += M4
    W2[2, 3] = 1 # C21 += M4
    
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C11 -= M5
    W2[1, 4] = 1 # C12 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11 += M7
    
    # 3. Combine U3, V3, W3 with U2, V2, W2
    # Indices:
    # I = i1*2 + i2
    # J = j1*2 + j2
    # K = k1*2 + k2
    # R = r3*7 + r2
    
    for i1 in range(3):
        for i2 in range(2):
            I = i1 * 2 + i2
            for j1 in range(3):
                for j2 in range(2):
                    J = j1 * 2 + j2
                    for r3_idx in range(rank_3):
                        u3_val = U3[i1 * 3 + j1, r3_idx]
                        if u3_val == 0:
                            continue
                        for r2_idx in range(rank_2):
                            u2_val = U2[i2 * 2 + j2, r2_idx]
                            if u2_val == 0:
                                continue
                            R = r3_idx * rank_2 + r2_idx
                            U[I, R] = u3_val * u2_val
                            
    for j1 in range(3):
        for j2 in range(2):
            J = j1 * 2 + j2
            for k1 in range(3):
                for k2 in range(2):
                    K = k1 * 2 + k2
                    for r3_idx in range(rank_3):
                        v3_val = V3[j1 * 3 + k1, r3_idx]
                        if v3_val == 0:
                            continue
                        for r2_idx in range(rank_2):
                            v2_val = V2[j2 * 2 + k2, r2_idx]
                            if v2_val == 0:
                                continue
                            R = r3_idx * rank_2 + r2_idx
                            V[J, R] = v3_val * v2_val
                            
    for k1 in range(3):
        for k2 in range(2):
            K = k1 * 2 + k2
            for i1 in range(3):
                for i2 in range(2):
                    I = i1 * 2 + i2
                    for r3_idx in range(rank_3):
                        w3_val = W3[k1 * 3 + i1, r3_idx]
                        if w3_val == 0:
                            continue
                        for r2_idx in range(rank_2):
                            w2_val = W2[k2 * 2 + i2, r2_idx]
                            if w2_val == 0:
                                continue
                            R = r3_idx * rank_2 + r2_idx
                            W[K, R] = w3_val * w2_val

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
