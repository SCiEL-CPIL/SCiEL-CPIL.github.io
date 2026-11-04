import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Construct M_2 (Strassen) and M_3 (Schoolbook)
    # M_2: n=2, rank=7
    # M_3: n=3, rank=27
    
    # Strassen M_2 factors
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[2, 0] = 1; W2[1, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[1, 5] = -1; W2[3, 5] = -1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[2, 6] = 1
    
    # Schoolbook M_3 factors
    # U3, V3, W3 are 9x27 matrices
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
                
    # Combine M_3 and M_2 to form M_6
    # Rank = 27 * 7 = 189
    rank = 189
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r3 in range(27):
        for r2 in range(7):
            r = r3 * 7 + r2
            
            # U6
            for i1 in range(3):
                for j1 in range(3):
                    if U3[i1*3+j1, r3] != 0:
                        for i2 in range(2):
                            for j2 in range(2):
                                if U2[i2*2+j2, r2] != 0:
                                    i = i1*2 + i2
                                    j = j1*2 + j2
                                    U6[i*6+j, r] += U3[i1*3+j1, r3] * U2[i2*2+j2, r2]
                                    
            # V6
            for j1 in range(3):
                for k1 in range(3):
                    if V3[j1*3+k1, r3] != 0:
                        for j2 in range(2):
                            for k2 in range(2):
                                if V2[j2*2+k2, r2] != 0:
                                    j = j1*2 + j2
                                    k = k1*2 + k2
                                    V6[j*6+k, r] += V3[j1*3+k1, r3] * V2[j2*2+k2, r2]
                                    
            # W6
            for k1 in range(3):
                for i1 in range(3):
                    if W3[k1*3+i1, r3] != 0:
                        for k2 in range(2):
                            for i2 in range(2):
                                if W2[k2*2+i2, r2] != 0:
                                    k = k1*2 + k2
                                    i = i1*2 + i2
                                    W6[k*6+i, r] += W3[k1*3+i1, r3] * W2[k2*2+i2, r2]
                                    
    if time.time() > deadline:
        # Fallback to schoolbook if too slow (unlikely)
        rank_fb = 216
        U_fb = np.zeros((36, rank_fb), dtype=int)
        V_fb = np.zeros((36, rank_fb), dtype=int)
        W_fb = np.zeros((36, rank_fb), dtype=int)
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_fb[i*6+j, col] = 1
                    V_fb[j*6+k, col] = 1
                    W_fb[k*6+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
