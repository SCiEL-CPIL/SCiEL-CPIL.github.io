DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # 2x2 Strassen factors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    
    # M2: (A10+A11)B00
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[1][1]=1; W2[3][1]=-1
    
    # M3: A00(B01-B11)
    U2[0][2]=1
    V2[2][2]=1; V2[3][2]=-1
    W2[2][2]=1; W2[3][2]=1
    
    # M4: A11(B10-B00)
    U2[3][3]=1
    V2[1][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[1][3]=1
    
    # M5: (A00+A01)B11
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[2][4]=1
    
    # M6: (A10-A00)(B00+B01)
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[2][5]=1
    W2[3][5]=1
    
    # M7: (A01-A11)(B10+B11)
    U2[1][6]=1; U2[3][6]=-1
    V2[1][6]=1; V2[3][6]=1
    W2[0][6]=1
    
    # 4x4 construction
    U4 = [[0]*rank for _ in range(16)]
    V4 = [[0]*rank for _ in range(16)]
    W4 = [[0]*rank for _ in range(16)]
    
    top_A = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    top_B = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    top_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    block_to_bi_bj = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
    
    for t in range(7):
        for s in range(7):
            r = t*7 + s
            for blk, coeff in top_A[t].items():
                bi, bj = block_to_bi_bj[blk]
                for li in range(2):
                    for lj in range(2):
                        u2_val = U2[li*2+lj][s]
                        if u2_val != 0:
                            i_g = 2*bi + li
                            j_g = 2*bj + lj
                            U4[i_g*4 + j_g][r] += coeff * u2_val
            for blk, coeff in top_B[t].items():
                bi, bj = block_to_bi_bj[blk]
                for li in range(2):
                    for lj in range(2):
                        v2_val = V2[li*2+lj][s]
                        if v2_val != 0:
                            j_g = 2*bi + li
                            k_g = 2*bj + lj
                            V4[j_g*4 + k_g][r] += coeff * v2_val
            for blk, coeff in top_W[t].items():
                bi, bj = block_to_bi_bj[blk]
                for li in range(2):
                    for lj in range(2):
                        w2_val = W2[lj*2+li][s]
                        if w2_val != 0:
                            k_g = 2*bj + lj
                            i_g = 2*bi + li
                            W4[k_g*4 + i_g][r] += coeff * w2_val
    
    return {"U": U4, "V": V4, "W": W4}
