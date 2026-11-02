import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    # Rank 65 construction
    # 49 for 4x4x4, 16 for 4x4x1
    
    # Strassen 2x2x2
    # U, V, W are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # Term 1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C10
    
    # Term 2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C01
    
    # Term 3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1 # C10
    
    # Term 4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1 # C01
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # Tensor product to get 4x4x4 (Rank 49)
    # U4 shape 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            idx = i * 4 + j
                            U4[idx, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            V4[idx, r] = V2[j1 * 2 + i1, r1] * V2[j2 * 2 + i2, r2] # Note: V index is j*p+k. Here p=2, k is next block row?
                            # Wait, V index for 2x2 is j*2+k.
                            # In tensor product, V_{j,k} = V_{j1,k1} V_{j2,k2}.
                            # j = j1*2+j2, k = k1*2+k2.
                            # So V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2].
                            # My loop above only iterates i,j. I need to iterate j,k for V and k,i for W.
                            
    # Redo V4 and W4 correctly
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            idx = j * 4 + k
                            V4[idx, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            idx = k * 4 + i
                            W4[idx, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]

    # 4x4x1 (Rank 16)
    # U41 shape 16x16, V41 shape 20x16 (only col 4 of B active), W41 shape 20x16 (only row 4 of C active)
    # Actually V is m*p = 4*5 = 20. W is p*n = 5*4 = 20.
    # For k=4 (index 4 in p), V[j*5+4, r] and W[4*4+i, r] are active.
    U41 = np.zeros((16, 16))
    V41 = np.zeros((20, 16))
    W41 = np.zeros((20, 16))
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U41[i * 4 + j, r] = 1.0
            V41[j * 5 + 4, r] = 1.0
            W41[4 * 4 + i, r] = 1.0

    # Combine
    Rank = 49 + 16
    U = np.zeros((16, Rank))
    V = np.zeros((20, Rank))
    W = np.zeros((20, Rank))
    
    U[:, :49] = U4
    V[:, :49] = np.zeros((20, 49))
    V[:16, :49] = V4 # V4 is 16x49, corresponds to k=0..3
    # V4 indices j*4+k. k is 0..3.
    # In global V, index is j*5+k.
    # So V[j*5+k, r] = V4[j*4+k, r] for k<4.
    # My V4 loop: idx = j*4+k.
    # So V_global[j*5+k, r] = V4[j*4+k, r].
    
    W[:, :49] = np.zeros((20, 49))
    W[:16, :49] = W4 # W4 is 16x49, corresponds to k=0..3
    # W4 indices k*4+i.
    # Global W index k*4+i.
    # So W_global[k*4+i, r] = W4[k*4+i, r].
    
    U[:, 49:] = U41
    V[:, 49:] = V41
    W[:, 49:] = W41

    # Convert to lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
