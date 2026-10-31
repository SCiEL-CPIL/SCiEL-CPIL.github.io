import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # 2x2 Strassen decomposition (rank 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1 = (a1+a4)(b1+b4)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # m2 = (a3+a4)b1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # m3 = a1(b2-b4)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # m4 = a4(b3-b1)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # m5 = (a1+a2)b4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # m6 = (a3-a1)(b1+b2)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # m7 = (a2-a4)(b3+b4)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Build 4x4 via recursive Strassen (rank 49)
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block index: block = br*2 + bc, where br,bc in {0,1}
    # Big term A-side: dict {block: coeff}
    big_A = [
        {0: 1, 3: 1},      # M1: A00+A11
        {2: 1, 3: 1},      # M2: A10+A11
        {0: 1},            # M3: A00
        {3: 1},            # M4: A11
        {0: 1, 1: 1},      # M5: A00+A01
        {2: 1, 0: -1},     # M6: A10-A00
        {1: 1, 3: -1},     # M7: A01-A11
    ]
    
    big_B = [
        {0: 1, 3: 1},      # M1: B00+B11
        {0: 1},            # M2: B00
        {1: 1, 3: -1},     # M3: B01-B11
        {2: 1, 0: -1},     # M4: B10-B00
        {3: 1},            # M5: B11
        {0: 1, 1: 1},      # M6: B00+B01
        {2: 1, 3: 1},      # M7: B10+B11
    ]
    
    big_C = [
        {0: 1, 3: 1},      # M1: C00(+1), C11(+1)
        {2: 1, 3: -1},     # M2: C10(+1), C11(-1)
        {1: 1, 3: 1},      # M3: C01(+1), C11(+1)
        {0: 1, 2: 1},      # M4: C00(+1), C10(+1)
        {0: -1, 1: 1},     # M5: C00(-1), C01(+1)
        {3: 1},            # M6: C11(+1)
        {0: 1, 3: 1},      # M7: C00(+1), C11(+1)
    ]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            for block, coeff_a in big_A[t].items():
                br = block // 2
                bc = block % 2
                for ar in range(2):
                    for ac in range(2):
                        u2_val = U2[ar*2+ac][s]
                        if u2_val != 0:
                            global_row = (br*2+ar)*4 + (bc*2+ac)
                            U[global_row][col] += coeff_a * u2_val
            
            for block, coeff_b in big_B[t].items():
                br = block // 2
                bc = block % 2
                for ar in range(2):
                    for ac in range(2):
                        v2_val = V2[ar*2+ac][s]
                        if v2_val != 0:
                            global_row = (br*2+ar)*4 + (bc*2+ac)
                            V[global_row][col] += coeff_b * v2_val
            
            for block, coeff_c in big_C[t].items():
                br = block // 2
                bc = block % 2
                for ar in range(2):
                    for ac in range(2):
                        w2_val = W2[ar*2+ac][s]
                        if w2_val != 0:
                            global_row = (bc*2+ac)*4 + (br*2+ar)
                            W[global_row][col] += coeff_c * w2_val
    
    # Verify with deadline check
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        # Quick verification
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    for j2 in range(4):
                        expected = 1 if j == j2 else 0
                        actual = 0
                        for s in range(49):
                            actual += U[i*4+j][s] * V[j2*4+k][s] * W[k*4+i][s]
                        if actual != expected:
                            # Fallback to schoolbook
                            rank_fb = 64
                            U_fb = [[0]*rank_fb for _ in range(16)]
                            V_fb = [[0]*rank_fb for _ in range(16)]
                            W_fb = [[0]*rank_fb for _ in range(16)]
                            col = 0
                            for ii in range(4):
                                for jj in range(4):
                                    for kk in range(4):
                                        U_fb[ii*4+jj][col] = 1
                                        V_fb[jj*4+kk][col] = 1
                                        W_fb[kk*4+ii][col] = 1
                                        col += 1
                            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
