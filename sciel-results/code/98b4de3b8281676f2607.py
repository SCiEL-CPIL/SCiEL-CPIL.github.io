import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    
    # Strassen A-combinations (4-vectors for 2x2 matrix)
    S_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    for k in [0, 2, 4, 6, 8]:
        k2 = k + 1
        # V vectors (22-dim)
        V1 = [0]*22; V1[0*p+k] = 1; V1[1*p+k2] = 1
        V2 = [0]*22; V2[0*p+k] = 1
        V3 = [0]*22; V3[1*p+k] = 1; V3[1*p+k2] = -1
        V4 = [0]*22; V4[0*p+k2] = 1; V4[1*p+k] = -1
        V5 = [0]*22; V5[1*p+k2] = 1
        V6 = [0]*22; V6[0*p+k] = 1; V6[1*p+k] = 1
        V7 = [0]*22; V7[0*p+k2] = 1; V7[1*p+k2] = 1
        
        # W vectors (22-dim), indexed by (k_col, i)
        W1 = [0]*22; W1[k*n+0] = 1; W1[k2*n+1] = 1
        W2 = [0]*22; W2[k*n+1] = 1; W2[k2*n+1] = -1
        W3 = [0]*22; W3[k2*n+0] = 1; W3[k2*n+1] = 1
        W4 = [0]*22; W4[k*n+0] = 1; W4[k*n+1] = 1
        W5 = [0]*22; W5[k*n+0] = -1; W5[k2*n+0] = 1
        W6 = [0]*22; W6[k2*n+1] = 1
        W7 = [0]*22; W7[k*n+0] = 1
        
        for t in range(7):
            U_cols.append(S_A[t][:])
            V_cols.append([V1,V2,V3,V4,V5,V6,V7][t][:])
            W_cols.append([W1,W2,W3,W4,W5,W6,W7][t][:])
    
    # Single column 10: schoolbook 4 terms
    for i in range(n):
        for j in range(m):
            u = [0]*4
            u[i*m+j] = 1
            v = [0]*22
            v[j*p+10] = 1
            w = [0]*22
            w[10*n+i] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
