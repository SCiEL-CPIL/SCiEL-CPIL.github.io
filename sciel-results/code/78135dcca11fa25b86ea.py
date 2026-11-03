import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen factors
    # U2, V2, W2 are 4x7 matrices
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=1: m1 = (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=2: m2 = (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # r=3: m3 = a00(b01-b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # r=4: m4 = a11(b10-b00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # r=5: m5 = (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # r=6: m6 = (a10-a00)(b00+b01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=7: m7 = (a01-a11)(b10+b11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Kronecker product for 4x4
    # U4 = U2 kron U2 ?
    # Indices: i = 2*i1 + i0, j = 2*j1 + j0
    # U4[i*4+j, r1*7+r0] = U2[i1*2+j1, r1] * U2[i0*2+j0, r0]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    idx_u = i*4 + j
                    idx_u2_hi = i1*2 + j1
                    idx_u2_lo = i0*2 + j0
                    for r1 in range(7):
                        u_hi = U2[idx_u2_hi, r1]
                        if u_hi == 0: continue
                        for r0 in range(7):
                            u_lo = U2[idx_u2_lo, r0]
                            if u_lo == 0: continue
                            val = u_hi * u_lo
                            idx_r = r1*7 + r0
                            U4[idx_u, idx_r] = val
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    idx_v = j*4 + k
                    idx_v2_hi = j1*2 + k1
                    idx_v2_lo = j0*2 + k0
                    for r1 in range(7):
                        v_hi = V2[idx_v2_hi, r1]
                        if v_hi == 0: continue
                        for r0 in range(7):
                            v_lo = V2[idx_v2_lo, r0]
                            if v_lo == 0: continue
                            val = v_hi * v_lo
                            idx_r = r1*7 + r0
                            V4[idx_v, idx_r] = val
                            
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    idx_w = k*4 + i
                    idx_w2_hi = k1*2 + i1
                    idx_w2_lo = k0*2 + i0
                    for r1 in range(7):
                        w_hi = W2[idx_w2_hi, r1]
                        if w_hi == 0: continue
                        for r0 in range(7):
                            w_lo = W2[idx_w2_lo, r0]
                            if w_lo == 0: continue
                            val = w_hi * w_lo
                            idx_r = r1*7 + r0
                            W4[idx_w, idx_r] = val

    # Convert to lists of lists as required by interface (integer/half-integer)
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
