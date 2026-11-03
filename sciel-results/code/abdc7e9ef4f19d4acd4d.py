DIMENSIONS = [4, 4, 9]

def construct():
    import time
    start_time = time.time()
    
    # Strassen 2x2x2 factors
    U2 = [[1,0,1,0,1,-1,0],
          [0,0,0,0,1,0,1],
          [0,1,0,0,0,1,0],
          [1,1,0,1,0,0,-1]]
    V2 = [[1,1,0,-1,0,1,0],
          [0,0,1,0,0,1,0],
          [0,0,0,1,0,0,1],
          [1,0,-1,0,1,0,1]]
    W2 = [[1,0,0,1,-1,0,1],
          [0,1,0,1,0,0,0],
          [0,0,1,0,1,0,0],
          [1,-1,1,0,0,1,0]]
    
    # Kronecker product to get 4x4x4 factors
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = i1*2 + i2
                    j = j1*2 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            U4[i*4+j][c] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = j1*2 + j2
                    k = k1*2 + k2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            V4[j*4+k][c] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = k1*2 + k2
                    i = i1*2 + i2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = c1*7 + c2
                            W4[k*4+i][c] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    # Full factors for 4x4 x 4x9
    n, m, p = 4, 4, 9
    rank = 49 + 49 + 4
    U = [[0]*rank for _ in range(n*m)]  # (16, 102)
    V = [[0]*rank for _ in range(m*p)]  # (36, 102)
    W = [[0]*rank for _ in range(p*n)]  # (36, 102)
    
    # Part 1: C1 = A*B1 (4x4 x 4x4 = 4x4), columns 0-48
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+k][c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
    
    # Part 2: C2 = A*B2 (4x4 x 4x4 = 4x4), columns 49-97
    for i in range(16):
        for c in range(49):
            U[i][49+c] = U4[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*9+(4+k)][49+c] = V4[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(4+k)*4+i][49+c] = W4[k*4+i][c]
    
    # Part 3: C3 = A*B3 (4x4 x 4x1 = 4x1), columns 98-101
    for i in range(4):
        for j in range(4):
            for c in range(4):
                U[i*4+j][98+c] = 1
    for j in range(4):
        for c in range(4):
            V[j*9+8][98+c] = 0.5
    for i in range(4):
        for c in range(4):
            W[8*4+i][98+c] = 0.5
    
    if time.time() - start_time > 110:
        pass  # Should be fast enough
    
    return {"U": U, "V": V, "W": W}
