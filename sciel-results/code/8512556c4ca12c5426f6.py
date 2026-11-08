import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    # Strassen 2x2x2 factors
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # P1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # C00 += P1, C11 += P1? 
    # Wait, W_s maps to C indices.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's fill W_s based on C formulas
    # W_s[k*2+i, r]
    # r=0 (P1): C00+=1, C11+=1
    W_s[0, 0] = 1
    W_s[3, 0] = 1
    # r=1 (P2): C10+=1, C11+=-1
    W_s[2, 1] = 1
    W_s[3, 1] = -1
    # r=2 (P3): C01+=1, C11+=1
    W_s[1, 2] = 1
    W_s[3, 2] = 1
    # r=3 (P4): C00+=1, C10+=1
    W_s[0, 3] = 1
    W_s[2, 3] = 1
    # r=4 (P5): C00+=-1, C01+=1
    W_s[0, 4] = -1
    W_s[1, 4] = 1
    # r=5 (P6): C11+=1
    W_s[3, 5] = 1
    # r=6 (P7): C00+=1
    W_s[0, 6] = 1

    # Now fill U_s and V_s
    # P1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    
    # P2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    
    # P3 = A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    
    # P4 = A11(B01-B00)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    
    # P5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    
    # P6 = (A10-A00)(B00+B10)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    
    # P7 = (A01-A11)(B01+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1

    # Schoolbook 1x2x3 factors
    # U_l: 2x6, V_l: 6x6, W_l: 3x6
    U_l = np.zeros((2, 6), dtype=int)
    V_l = np.zeros((6, 6), dtype=int)
    W_l = np.zeros((3, 6), dtype=int)
    
    # C_k = A0 B_0k + A1 B_1k
    # Term 0: A0 B_00 -> C0
    U_l[0, 0] = 1
    V_l[0, 0] = 1 # j=0, k=0 -> idx 0
    W_l[0, 0] = 1
    
    # Term 1: A0 B_01 -> C1
    U_l[0, 1] = 1
    V_l[1, 1] = 1 # j=0, k=1 -> idx 1
    W_l[1, 1] = 1
    
    # Term 2: A0 B_02 -> C2
    U_l[0, 2] = 1
    V_l[2, 2] = 1 # j=0, k=2 -> idx 2
    W_l[2, 2] = 1
    
    # Term 3: A1 B_10 -> C0
    U_l[1, 3] = 1
    V_l[3, 3] = 1 # j=1, k=0 -> idx 3
    W_l[0, 3] = 1
    
    # Term 4: A1 B_11 -> C1
    U_l[1, 4] = 1
    V_l[4, 4] = 1 # j=1, k=1 -> idx 4
    W_l[1, 4] = 1
    
    # Term 5: A1 B_12 -> C2
    U_l[1, 5] = 1
    V_l[5, 5] = 1 # j=1, k=2 -> idx 5
    W_l[2, 5] = 1

    # Kronecker Product
    U = np.zeros((8, 42), dtype=int)
    V = np.zeros((24, 42), dtype=int)
    W = np.zeros((12, 42), dtype=int)
    
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2
            
            # U
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        u_val = U_s[i1 * 2 + j1, r1] * U_l[j2, r2]
                        if u_val != 0:
                            idx_u = i1 * 4 + (2 * j1 + j2)
                            U[idx_u, r] = u_val
            
            # V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v_val = V_s[j1 * 2 + k1, r1] * V_l[j2 * 3 + k2, r2]
                            if v_val != 0:
                                idx_v = (2 * j1 + j2) * 6 + (3 * k1 + k2)
                                V[idx_v, r] = v_val
            
            # W
            for i1 in range(2):
                for k1 in range(2):
                    for k2 in range(3):
                        w_val = W_s[k1 * 2 + i1, r1] * W_l[k2, r2]
                        if w_val != 0:
                            idx_w = (3 * k1 + k2) * 2 + i1
                            W[idx_w, r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
