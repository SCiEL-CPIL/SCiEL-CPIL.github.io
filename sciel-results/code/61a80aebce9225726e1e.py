import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2x2 Strassen decomposition
    # U, V, W are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: (A00+A11)(B00+B11) -> C00(+), C11(+)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: (A10+A11)B00 -> C10(+), C11(-)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 3: A00(B01-B11) -> C01(+), C11(+)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 4: A11(B10-B00) -> C10(+), C00(-)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[0][3] = -1
    
    # Term 5: (A00+A01)B11 -> C00(-), C01(+)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6: (A10-A00)(B00+B01) -> C11(+)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C00(+)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 2. Construct 4x4x4 decomposition via Kronecker product
    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u_val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if u_val != 0:
                                U4[i*4+j][r] = u_val
                            
                            v_val = V2[j1*2+ (0) ][r1] # Wait, V2 index is j,k.
                            # V2 row index is j*2+k.
                            # In Kronecker, V index is (j1,k1) and (j2,k2).
                            # V4 row index is j*4+k where j=2j1+j2, k=2k1+k2.
                            # So V4[ (2j1+j2)*4 + (2k1+k2) ][r]
                            # V2[ j1*2+k1 ][r1] * V2[ j2*2+k2 ][r2]
                            
                            # Let's loop k1, k2
                            for k1 in range(2):
                                for k2 in range(2):
                                    j = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    v_val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                    if v_val != 0:
                                        V4[j*4+k][r] = v_val
                                        
                                    w_val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                    if w_val != 0:
                                        W4[k*4+i][r] = w_val

    if time.time() > deadline:
        return {"U": U4, "V": V4, "W": W4} # Fallback, though incomplete for 4x4x5

    # 3. Construct 4x4x1 decomposition
    rank_441 = 16
    U1 = [[0]*rank_441 for _ in range(16)]
    V1 = [[0]*rank_441 for _ in range(4)]
    W1 = [[0]*rank_441 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U1[i*4+j][r] = 1
            V1[j][r] = 1
            W1[i][r] = 1 # k=0 in local, but global k=4. W index is k*4+i.
            
    # 4. Combine
    total_rank = rank_444 + rank_441
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(20)] # m*p = 4*5 = 20
    W = [[0]*total_rank for _ in range(20)] # p*n = 5*4 = 20
    
    # Copy 4x4x4 part
    for i in range(16):
        for r in range(rank_444):
            U[i][r] = U4[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j*4+k][r] = V4[j*4+k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k*4+i][r] = W4[k*4+i][r]
                
    # Copy 4x4x1 part
    offset = rank_444
    for i in range(16):
        for r in range(rank_441):
            U[i][offset + r] = U1[i][r]
    for j in range(4):
        for r in range(rank_441):
            V[j*4 + 4][offset + r] = V1[j][r] # k=4
    for i in range(4):
        for r in range(rank_441):
            W[4*4 + i][offset + r] = W1[i][r] # k=4, so index 16+i

    return {"U": U, "V": V, "W": W}
