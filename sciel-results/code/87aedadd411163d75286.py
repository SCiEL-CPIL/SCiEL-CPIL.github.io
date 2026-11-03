import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # 1. Strassen 2x2x2 Factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i, j, c], V[j, k, c], W[k, i, c]
    # U rows: i*2+j (0..3), V rows: j*2+k (0..3), W rows: k*2+i (0..3)
    
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U_S[0][0] = 1; U_S[3][0] = 1
    V_S[0][0] = 1; V_S[3][0] = 1
    W_S[0][0] = 1; W_S[3][0] = 1
    
    # M2: (A10+A11)B00
    U_S[2][1] = 1; U_S[3][1] = 1
    V_S[0][1] = 1
    W_S[2][1] = 1
    
    # M3: A00(B01-B11)
    U_S[0][2] = 1
    V_S[1][2] = 1; V_S[3][2] = -1
    W_S[1][2] = 1
    
    # M4: A11(B10-B00)
    U_S[3][3] = 1
    V_S[2][3] = 1; V_S[0][3] = -1
    W_S[0][3] = 1; W_S[2][3] = 1
    
    # M5: (A00+A01)B11
    U_S[0][4] = 1; U_S[1][4] = 1
    V_S[3][4] = 1
    W_S[0][4] = -1; W_S[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_S[2][5] = 1; U_S[0][5] = -1
    V_S[0][5] = 1; V_S[1][5] = 1
    W_S[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_S[1][6] = 1; U_S[3][6] = -1
    V_S[2][6] = 1; V_S[3][6] = 1
    W_S[0][6] = 1; W_S[3][6] = -1
    
    # 2. Kronecker Product for 4x4x4 (Rank 49)
    # U_44 = U_S (x) U_S
    # Size: 16x49
    U_44 = [[0]*49 for _ in range(16)]
    V_44 = [[0]*49 for _ in range(16)]
    W_44 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for c1 in range(7):
                u1 = U_S[i1*2+j1][c1]
                if u1 != 0:
                    for i2 in range(2):
                        for j2 in range(2):
                            for c2 in range(7):
                                u2 = U_S[i2*2+j2][c2]
                                if u2 != 0:
                                    u_val = u1 * u2
                                    r_u = (i1*2+i2)*2 + (j1*2+j2)
                                    c_u = c1*7 + c2
                                    U_44[r_u][c_u] = u_val
                                    
    # V and W are similar
    for j1 in range(2):
        for k1 in range(2):
            for c1 in range(7):
                v1 = V_S[j1*2+k1][c1]
                if v1 != 0:
                    for j2 in range(2):
                        for k2 in range(2):
                            for c2 in range(7):
                                v2 = V_S[j2*2+k2][c2]
                                if v2 != 0:
                                    v_val = v1 * v2
                                    r_v = (j1*2+j2)*2 + (k1*2+k2)
                                    c_v = c1*7 + c2
                                    V_44[r_v][c_v] = v_val

    for k1 in range(2):
        for i1 in range(2):
            for c1 in range(7):
                w1 = W_S[k1*2+i1][c1]
                if w1 != 0:
                    for k2 in range(2):
                        for i2 in range(2):
                            for c2 in range(7):
                                w2 = W_S[k2*2+i2][c2]
                                if w2 != 0:
                                    w_val = w1 * w2
                                    r_w = (k1*2+k2)*2 + (i1*2+i2)
                                    c_w = c1*7 + c2
                                    W_44[r_w][c_w] = w_val

    # 3. Schoolbook 4x4x1 (Rank 16)
    # U_441: 16x16
    # V_441: 4x16
    # W_441: 4x16
    U_441 = [[0]*16 for _ in range(16)]
    V_441 = [[0]*16 for _ in range(4)]
    W_441 = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            c = i*4 + j
            U_441[i*4+j][c] = 1
            V_441[j][c] = 1
            W_441[i][c] = 1

    # 4. Assemble 4x4x5 (Rank 65)
    rank = 65
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # Copy 44 part
    for r in range(16):
        for c in range(49):
            U[r][c] = U_44[r][c]
            V[r][c] = V_44[r][c]
            W[r][c] = W_44[r][c]
            
    # Copy 41 part (cols 49..64)
    for r in range(16):
        for c in range(16):
            U[r][49+c] = U_441[r][c]
            
    for r in range(4):
        for c in range(16):
            V[16+r][49+c] = V_441[r][c]
            W[16+r][49+c] = W_441[r][c]

    return {"U": U, "V": V, "W": W}
