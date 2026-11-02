import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Strassen 2x2x2 decomposition (rank 7)
    # T1[i,j,k] = 1 for i,j,k in {0,1}
    # U1 shape (4,7), V1 shape (4,7), W1 shape (4,7)
    # U1[i*2+j, r], V1[j*2+k, r], W1[k*2+i, r]
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Strassen products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)(B00)
    # M2 = (A00)(B01-B11)
    # M3 = (A11)(B10-B00)
    # M4 = (A00+A01)(B11)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # U1: A-side, index i*2+j
    U1[0,0] = 1; U1[3,0] = 1   # M0: A00, A11
    U1[2,1] = 1; U1[3,1] = 1   # M1: A10, A11
    U1[0,2] = 1                # M2: A00
    U1[3,3] = 1                # M3: A11
    U1[0,4] = 1; U1[1,4] = 1   # M4: A00, A01
    U1[2,5] = 1; U1[0,5] = -1  # M5: A10, -A00
    U1[1,6] = 1; U1[3,6] = -1  # M6: A01, -A11
    
    # V1: B-side, index j*2+k
    V1[0,0] = 1; V1[3,0] = 1   # M0: B00, B11
    V1[0,1] = 1                # M1: B00
    V1[1,2] = 1; V1[3,2] = -1  # M2: B01, -B11
    V1[2,3] = 1; V1[0,3] = -1  # M3: B10, -B00
    V1[3,4] = 1                # M4: B11
    V1[0,5] = 1; V1[1,5] = 1   # M5: B00, B01
    V1[2,6] = 1; V1[3,6] = 1   # M6: B10, B11
    
    # W1: C-side, index k*2+i (trace convention)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # C[i,k] -> index k*2+i
    # C00: i=0,k=0 -> idx 0
    # C01: i=0,k=1 -> idx 2
    # C10: i=1,k=0 -> idx 1
    # C11: i=1,k=1 -> idx 3
    W1[0,0] = 1; W1[0,3] = 1; W1[0,4] = -1; W1[0,6] = 1  # C00
    W1[2,2] = 1; W1[2,4] = 1                              # C01
    W1[1,1] = 1; W1[1,3] = 1                              # C10
    W1[3,0] = 1; W1[3,1] = -1; W1[3,2] = 1; W1[3,5] = 1   # C11
    
    # Trivial 1x2x3 decomposition (rank 6)
    # T2[i,j,k] = 1 for i=0, j in {0,1}, k in {0,1,2}
    # U2 shape (2,6), V2 shape (6,6), W2 shape (3,6)
    # U2[j, r], V2[j*3+k, r], W2[k, r]
    
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    # 6 multiplications: A0*B00, A0*B01, A0*B02, A1*B10, A1*B11, A1*B12
    for t in range(6):
        j = t // 3  # 0 or 1
        k = t % 3   # 0, 1, or 2
        U2[j, t] = 1
        V2[j*3 + k, t] = 1
        W2[k, t] = 1
    
    # Kronecker product: T = T1 ⊗ T2
    # T1 shape (2,2,2), T2 shape (1,2,3)
    # Combined: n=2*1=2, m=2*2=4, p=2*3=6
    # Rank: 7*6 = 42
    
    # Index mappings:
    # A[i, a] where i in {0,1}, a in {0,1,2,3}
    #   i = i1 (from T1), a = j1*2 + j2 (j1 from T1, j2 from T2)
    #   U index: i*4 + a = i1*4 + j1*2 + j2
    #   
    # B[a, b] where a in {0,1,2,3}, b in {0,1,2,3,4,5}
    #   a = j1*2 + j2, b = k1*3 + k2
    #   V index: a*6 + b = (j1*2+j2)*6 + (k1*3+k2)
    #   
    # C[i, b] where i in {0,1}, b in {0,1,2,3,4,5}
    #   i = i1, b = k1*3 + k2
    #   W index: b*2 + i = (k1*3+k2)*2 + i1 (trace convention: k*n+i)
    
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((24, rank))
    W = np.zeros((12, rank))
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U: sum over i1, j1, i2, j2
            for i1 in range(2):
                for j1 in range(2):
                    v1 = U1[i1*2 + j1, r1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        v2 = U2[j2, r2]
                        if v2 == 0:
                            continue
                        idx_u = i1 * 4 + j1 * 2 + j2
                        U[idx_u, r] += v1 * v2
            
            # V: sum over j1, k1, j2, k2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1*2 + k1, r1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2 = V2[j2*3 + k2, r2]
                            if v2 == 0:
                                continue
                            a = j1 * 2 + j2
                            b = k1 * 3 + k2
                            idx_v = a * 6 + b
                            V[idx_v, r] += v1 * v2
            
            # W: sum over i1, k1, i2, k2
            for i1 in range(2):
                for k1 in range(2):
                    v1 = W1[k1*2 + i1, r1]
                    if v1 == 0:
                        continue
                    for k2 in range(3):
                        v2 = W2[k2, r2]
                        if v2 == 0:
                            continue
                        i = i1
                        b = k1 * 3 + k2
                        idx_w = b * 2 + i
                        W[idx_w, r] += v1 * v2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
