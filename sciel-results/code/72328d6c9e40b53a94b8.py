"""Matrix multiplication 6x6x6 using Kronecker product of Strassen (2x2, rank 7) and Schoolbook (3x3, rank 27)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    # Strassen 2x2 decomposition (Rank 7)
    # U2, V2, W2 shapes (4, 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1 # C00, C11? No, W2 index is k*2+i.
    # C00 (k=0,i=0) idx 0. C11 (k=1,i=1) idx 3.
    # M1 contributes to C00 and C11.
    W2[0,0]=1
    W2[3,0]=1
    
    # M2: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1 # C10 (k=0,i=1) idx 1
    
    # M3: A00(B10-B11)
    U2[0,2]=1
    V2[2,2]=1; V2[3,2]=-1
    W2[2,2]=1 # C01 (k=1,i=0) idx 2
    
    # M4: A11(B01-B00)
    U2[3,3]=1
    V2[1,3]=1; V2[0,3]=-1
    W2[0,3]=1 # C00
    W2[1,3]=1 # C10
    
    # M5: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1 # C00
    W2[2,4]=1 # C01
    
    # M6: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1 # C00
    
    # Schoolbook 3x3 decomposition (Rank 27)
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

    # Construct 6x6 decomposition
    R2, R3 = 7, 27
    R = R2 * R3
    U = np.zeros((36, R), dtype=int)
    V = np.zeros((36, R), dtype=int)
    W = np.zeros((36, R), dtype=int)
    
    for i in range(6):
        i2, i3 = divmod(i, 3)
        for j in range(6):
            j2, j3 = divmod(j, 3)
            u_idx = i * 6 + j
            for c2 in range(R2):
                u2_val = U2[i2*2+j2, c2]
                if u2_val == 0: continue
                for c3 in range(R3):
                    u3_val = U3[i3*3+j3, c3]
                    if u3_val == 0: continue
                    c = c2 * R3 + c3
                    U[u_idx, c] = u2_val * u3_val
    
    for j in range(6):
        j2, j3 = divmod(j, 3)
        for k in range(6):
            k2, k3 = divmod(k, 3)
            v_idx = j * 6 + k
            for c2 in range(R2):
                v2_val = V2[j2*2+k2, c2]
                if v2_val == 0: continue
                for c3 in range(R3):
                    v3_val = V3[j3*3+k3, c3]
                    if v3_val == 0: continue
                    c = c2 * R3 + c3
                    V[v_idx, c] = v2_val * v3_val
                    
    for k in range(6):
        k2, k3 = divmod(k, 3)
        for i in range(6):
            i2, i3 = divmod(i, 3)
            w_idx = k * 6 + i
            for c2 in range(R2):
                w2_val = W2[k2*2+i2, c2]
                if w2_val == 0: continue
                for c3 in range(R3):
                    w3_val = W3[k3*3+i3, c3]
                    if w3_val == 0: continue
                    c = c2 * R3 + c3
                    W[w_idx, c] = w2_val * w3_val
                    
    return {"U": U, "V": V, "W": W}
