import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition
    # Strassen 2x2x2
    # U_l, V_l, W_l are 4x7 matrices.
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    # M2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    # M3
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    # M4
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    # M5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    # M6
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    # M7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1
    
    R = 56
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    # Precompute indices
    a_maps = [
        [0, 1, 4, 5], # s=0: A[0,0], A[0,1], A[1,0], A[1,1]
        [2, 3, 6, 7]  # s=1: A[0,2], A[0,3], A[1,2], A[1,3]
    ]
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            a_idx = a_maps[s]
            
            # B indices
            if s == 0:
                b_idx = [2*j, 2*j+1, 8+2*j, 8+2*j+1]
            else:
                b_idx = [16+2*j, 16+2*j+1, 24+2*j, 24+2*j+1]
            
            # C indices
            # C(i, k) index = k*2 + i
            # k = 2*j, 2*j+1
            # i = 0, 1
            # Strassen W_local: [C00, C01, C10, C11]
            # C00 = C(0, 2j) = 4j
            # C01 = C(0, 2j+1) = 4j+2
            # C10 = C(1, 2j) = 4j+1
            # C11 = C(1, 2j+1) = 4j+3
            c_idx = [4*j, 4*j+2, 4*j+1, 4*j+3]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_s[la, l]
                    if val != 0:
                        U[a_idx[la], r] += val
                for lb in range(4):
                    val = V_s[lb, l]
                    if val != 0:
                        V[b_idx[lb], r] += val
                for lc in range(4):
                    val = W_s[lc, l]
                    if val != 0:
                        W[c_idx[lc], r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
