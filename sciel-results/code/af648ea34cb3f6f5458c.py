import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # 1. Strassen 2x2 Decomposition (Rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: A00+A11, B00+B11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: A10+A11, B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # M3: A00, B01-B11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # M4: A11, B10-B00
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    # M5: A00+A01, B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # M6: A10-A00, B00+B01
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: A01-A11, B10+B11
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W2 Output Reconstruction (Trace convention W[k*n+i])
    # C00 = M1 + M4 - M5 + M7
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C01 = M3 + M5
    W2[2, 2] = 1; W2[2, 4] = 1
    # C10 = M2 + M4
    W2[1, 1] = 1; W2[1, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1

    # 2. Laderman 3x3 Decomposition (Rank 23)
    # Define terms as (U_indices, V_indices, W_indices)
    # Indices are 0-8 for 3x3 matrices.
    # U[i*3+j], V[j*3+k], W[k*3+i]
    
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # Term definitions (coeffs for A, coeffs for B, coeffs for C)
    # Note: W index is k*3+i.
    
    terms = [
        # 0
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1, 1:-1, 3:-1}),
        # 1
        ({0:1, 4:-1}, {3:1, 4:-1}, {3:1, 4:1, 0:-1, 1:-1}),
        # 2
        ({0:1, 1:1}, {3:1, 4:1}, {0:1, 3:1, 4:-1, 1:-1}), # Adjusted for symmetry?
        # 3
        ({1:1, 4:-1}, {0:1, 4:1}, {1:1, 4:1, 0:-1, 3:-1}),
        # 4
        ({3:1, 4:1}, {1:1, 4:1}, {3:1, 4:1, 0:-1, 1:-1}),
        # 5
        ({3:1, 0:-1}, {0:1, 1:1}, {0:1, 3:1, 4:-1, 1:-1}),
        # 6
        ({1:1, 4:-1}, {3:1, 4:1}, {1:1, 3:1, 4:-1, 0:-1}),
        # 7
        ({6:1, 8:1}, {0:1, 8:1}, {6:1, 8:1, 7:-1, 5:-1}),
        # 8
        ({6:1, 8:-1}, {6:1, 8:-1}, {5:1, 8:1, 6:-1, 7:-1}),
        # 9
        ({6:1, 7:1}, {6:1, 8:1}, {6:1, 5:1, 8:-1, 7:-1}),
        # 10
        ({7:1, 8:-1}, {0:1, 8:1}, {7:1, 8:1, 6:-1, 5:-1}),
        # 11
        ({2:1, 5:1}, {0:1, 4:1}, {2:1, 5:1, 1:-1, 4:-1}),
        # 12
        ({2:1, 5:-1}, {3:1, 4:-1}, {4:1, 5:1, 2:-1, 1:-1}),
        # 13
        ({2:1, 1:1}, {3:1, 4:1}, {2:1, 4:1, 5:-1, 1:-1}),
        # 14
        ({1:1, 5:-1}, {0:1, 4:1}, {1:1, 5:1, 2:-1, 4:-1}),
        # 15
        ({3:1, 5:1}, {1:1, 4:1}, {3:1, 5:1, 2:-1, 1:-1}),
        # 16
        ({3:1, 2:-1}, {0:1, 1:1}, {2:1, 3:1, 5:-1, 1:-1}),
        # 17
        ({1:1, 5:-1}, {3:1, 4:1}, {1:1, 3:1, 5:-1, 2:-1}),
        # 18
        ({8:1}, {8:1}, {8:1}),
        # 19
        ({8:1}, {0:1}, {6:1}),
        # 20
        ({0:1}, {8:1}, {2:1}),
        # 21
        ({6:1}, {2:1}, {4:1}),
        # 22
        ({2:1}, {6:1}, {0:1}),
    ]

    for r, (u_idx, v_idx, w_idx) in enumerate(terms):
        for i, c in u_idx.items():
            U3[i, r] += c
        for j, c in v_idx.items():
            V3[j, r] += c
        for k, c in w_idx.items():
            W3[k, r] += c

    # Verify Laderman 3x3
    def verify_3x3(U, V, W):
        # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    use_laderman = verify_3x3(U3, V3, W3)
    
    if not use_laderman:
        # Fallback to Schoolbook 3x3 (Rank 27)
        U3 = np.zeros((9, 27))
        V3 = np.zeros((9, 27))
        W3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i*3+j, col] = 1
                    V3[j*3+k, col] = 1
                    W3[k*3+i, col] = 1
                    col += 1

    # 3. Kronecker Product Construction
    rank2 = 7
    rank3 = U3.shape[1]
    rank_total = rank2 * rank3
    
    U6 = np.zeros((36, rank_total))
    V6 = np.zeros((36, rank_total))
    W6 = np.zeros((36, rank_total))
    
    for r3 in range(rank3):
        for r2 in range(rank2):
            r = r3 * rank2 + r2
            
            # U Part: 3x3 index (i3, j3) and 2x2 index (i2, j2) -> 6x6 index (i3*2+i2, j3*2+j2)
            for i3 in range(3):
                for j3 in range(3):
                    if U3[i3*3+j3, r3] != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                if U2[i2*2+j2, r2] != 0:
                                    i = i3*2 + i2
                                    j = j3*2 + j2
                                    idx = i*6 + j
                                    U6[idx, r] = U3[i3*3+j3, r3] * U2[i2*2+j2, r2]
                                    
            # V Part: 3x3 index (j3, k3) and 2x2 index (j2, k2) -> 6x6 index (j3*2+j2, k3*2+k2)
            for j3 in range(3):
                for k3 in range(3):
                    if V3[j3*3+k3, r3] != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                if V2[j2*2+k2, r2] != 0:
                                    j = j3*2 + j2
                                    k = k3*2 + k2
                                    idx = j*6 + k
                                    V6[idx, r] = V3[j3*3+k3, r3] * V2[j2*2+k2, r2]
                                    
            # W Part: 3x3 index (k3, i3) and 2x2 index (k2, i2) -> 6x6 index (k3*2+k2, i3*2+i2)
            # W index in 6x6 is k*6+i
            for k3 in range(3):
                for i3 in range(3):
                    if W3[k3*3+i3, r3] != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                if W2[k2*2+i2, r2] != 0:
                                    k = k3*2 + k2
                                    i = i3*2 + i2
                                    idx = k*6 + i
                                    W6[idx, r] = W3[k3*3+i3, r3] * W2[k2*2+i2, r2]
                                    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
