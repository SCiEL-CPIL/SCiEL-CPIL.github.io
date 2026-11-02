import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_222():
    # Returns U, V, W for 2x2x2
    # n=2, m=2, p=2
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    # U1: n1*m1 x r1, U2: n2*m2 x r2
    # Result U: (n1*n2)*(m1*m2) x (r1*r2)
    # Index mapping:
    # i1 in [0, n1), j1 in [0, m1), i2 in [0, n2), j2 in [0, m2)
    # i_tot = i1*n2 + i2
    # j_tot = j1*m2 + j2
    # r_tot = r1*r2 + r2
    # U1 index: i1*m1 + j1
    # U2 index: i2*m2 + j2
    # This mapping is tricky because U1 is flattened i,j.
    # We need to iterate over all entries.
    
    n1, m1, r1 = U1.shape[0] // 2, U1.shape[0] // 2, U1.shape[1] # Assuming square blocks for simplicity in logic, but shapes are given
    # Actually, let's just use the shapes.
    # U1 shape: (n1*m1, r1)
    # U2 shape: (n2*m2, r2)
    
    # We need n1, m1, n2, m2.
    # For 2x2x2, n=2, m=2.
    # For 1x1x2, n=1, m=1.
    
    # Let's hardcode the dimensions for the specific calls.
    pass

def construct():
    # 1. Get Strassen 2x2x2
    U2, V2, W2 = _strassen_222()
    
    # 2. Tensor product 2x2x2 with 2x2x2 to get 4x4x4
    # n=4, m=4, p=4
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    
    # We can construct this by iterating over terms.
    # Term (r1, r2)
    # U[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
    # i_tot = i1*2 + i2
    # j_tot = j1*2 + j2
    # U_tot[i_tot*4 + j_tot, r1*7 + r2] += U2_val * U22_val
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2 + j1, r1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2 + j2, r2]
                            if u2 == 0: continue
                            i_tot = i1 * 2 + i2
                            j_tot = j1 * 2 + j2
                            U4[i_tot*4 + j_tot, r] += u1 * u2
                            
                            v1 = V2[i1*2 + j1, r1] # Wait, V indices are j, k.
                            # For V, i1 is j1, j1 is k1?
                            # V2 shape 4x7. Index j*p + k. p=2.
                            # So V2[j1*2 + k1, r1].
                            # In the loop above, I used i1, j1 for U.
                            # For V, the first index is j, second is k.
                            # So we need to loop over j1, k1 for V2.
                            # But U2 loop is over i1, j1.
                            # This mismatch is a problem.
                            # U2[i,j], V2[j,k], W2[k,i].
                            # In tensor product:
                            # U_tot(i1,i2, j1,j2) = U1(i1,j1) U2(i2,j2)
                            # V_tot(j1,j2, k1,k2) = V1(j1,k1) V2(j2,k2)
                            # W_tot(k1,k2, i1,i2) = W1(k1,i1) W2(k2,i2)
                            
                            # My loop structure was only for U.
                            # I need to handle V and W separately or carefully.
                            
    # Let's rewrite the tensor product logic properly.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2 + j1, r1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2 + j2, r2]
                            if u2 == 0: continue
                            i_tot = i1 * 2 + i2
                            j_tot = j1 * 2 + j2
                            U4[i_tot*4 + j_tot, r] = u1 * u2
                            
            # V
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2 + k1, r1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V2[j2*2 + k2, r2]
                            if v2 == 0: continue
                            j_tot = j1 * 2 + j2
                            k_tot = k1 * 2 + k2
                            V4[j_tot*4 + k_tot, r] = v1 * v2
                            
            # W
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2 + i1, r1]
                    if w1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W2[k2*2 + i2, r2]
                            if w2 == 0: continue
                            k_tot = k1 * 2 + k2
                            i_tot = i1 * 2 + i2
                            W4[k_tot*4 + i_tot, r] = w1 * w2

    # 3. Tensor product 4x4x4 with 1x1x2 to get 4x4x8
    # n=4, m=4, p=8
    # U: 16 x 49
    # V: 32 x 49
    # W: 32 x 49
    
    # M_112:
    U1 = np.ones((1, 1))
    V1 = np.ones((1, 2)) # j=0, k=0,1
    W1 = np.ones((2, 1)) # k=0,1, i=0
    
    U8 = np.zeros((16, 49))
    V8 = np.zeros((32, 49))
    W8 = np.zeros((32, 49))
    
    for r in range(49):
        # U
        for i in range(4):
            for j in range(4):
                u4 = U4[i*4 + j, r]
                if u4 == 0: continue
                u1 = U1[0, 0]
                i_tot = i
                j_tot = j
                U8[i_tot*4 + j_tot, r] = u4 * u1
                
        # V
        for j in range(4):
            for k4 in range(4):
                v4 = V4[j*4 + k4, r]
                if v4 == 0: continue
                for k1 in range(2):
                    v1 = V1[0, k1]
                    j_tot = j
                    k_tot = k4 * 2 + k1
                    V8[j_tot*8 + k_tot, r] = v4 * v1
                    
        # W
        for k4 in range(4):
            for i in range(4):
                w4 = W4[k4*4 + i, r]
                if w4 == 0: continue
                for k1 in range(2):
                    w1 = W1[k1, 0]
                    k_tot = k4 * 2 + k1
                    i_tot = i
                    W8[k_tot*4 + i_tot, r] = w4 * w1

    # 4. Slice to 4x4x5
    # p=5
    # V: 20 x 49
    # W: 20 x 49
    
    V5 = np.zeros((20, 49))
    W5 = np.zeros((20, 49))
    
    # V8 rows: j*8 + k_tot. We need k_tot in [0, 5).
    for j in range(4):
        for k_tot in range(5):
            V5[j*5 + k_tot, :] = V8[j*8 + k_tot, :]
            
    # W8 rows: k_tot*4 + i. We need k_tot in [0, 5).
    for k_tot in range(5):
        for i in range(4):
            W5[k_tot*4 + i, :] = W8[k_tot*4 + i, :]
            
    U5 = U8
    
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
