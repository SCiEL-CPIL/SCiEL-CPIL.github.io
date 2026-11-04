import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1: (a00+a11)(b00+b11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (a10+a11)b00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 = M2+M4, C11 = M1-M2+...
    
    # M3: a00(b01-b11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01 = M3+M5, C11 = ...+M3+...
    
    # M4: a11(b10-b00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00 = M1+M4-..., C10 = M2+M4
    
    # M5: (a00+a01)b11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 = ...-M5, C01 = ...+M5
    
    # M6: (a10-a00)(b00+b01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 = ...+M6
    
    # M7: (a01-a11)(b10+b11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 = ...+M7
    
    rank = 27 * 7
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    for t3 in range(27):
        i3 = t3 // 9
        j3 = (t3 // 3) % 3
        k3 = t3 % 3
        
        for t2 in range(7):
            col = t3 * 7 + t2
            
            # Iterate over 2x2 indices
            for i2 in range(2):
                for j2 in range(2):
                    u_val = U2[2*i2 + j2, t2]
                    if u_val == 0: continue
                    
                    for k2 in range(2):
                        v_val = V2[2*j2 + k2, t2]
                        w_val = W2[2*k2 + i2, t2]
                        
                        if v_val == 0 or w_val == 0: continue
                        
                        row_u = 12*i3 + 6*i2 + 2*j3 + j2
                        row_v = 12*j3 + 6*j2 + 2*k3 + k2
                        row_w = 12*k3 + 6*k2 + 2*i3 + i2
                        
                        U[row_u, col] = u_val
                        V[row_v, col] = v_val
                        W[row_w, col] = w_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
