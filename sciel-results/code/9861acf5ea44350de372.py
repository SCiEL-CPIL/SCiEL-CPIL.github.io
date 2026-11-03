import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = 6, 6, 6
    
    # Strassen 2x2 factors (rank 7)
    U1 = [[1,0,0,1],
          [0,0,1,1],
          [1,0,0,0],
          [0,0,0,1],
          [1,1,0,0],
          [-1,0,1,0],
          [0,1,0,-1]]
    
    V1 = [[1,0,0,1],
          [1,0,0,0],
          [0,1,0,-1],
          [-1,0,1,0],
          [0,0,0,1],
          [1,1,0,0],
          [0,0,1,1]]
    
    W1 = [[1,0,0,1,-1,0,1],
          [0,0,1,0,1,0,0],
          [0,1,0,1,0,0,0],
          [1,-1,1,0,0,1,0]]
    
    # Schoolbook 3x3 factors (rank 27)
    r2_max = 27
    U2 = [[0]*r2_max for _ in range(9)]
    V2 = [[0]*r2_max for _ in range(9)]
    W2 = [[0]*r2_max for _ in range(9)]
    for i2 in range(3):
        for j2 in range(3):
            for k2 in range(3):
                r2 = i2*9 + j2*3 + k2
                U2[i2*3+j2][r2] = 1
                V2[j2*3+k2][r2] = 1
                W2[k2*3+i2][r2] = 1
    
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(3):
                    for j2 in range(3):
                        for k2 in range(3):
                            a_row = (i1*3+i2)*6 + (j1*3+j2)
                            b_row = (j1*3+j2)*6 + (k1*3+k2)
                            c_row = (k1*3+k2)*6 + (i1*3+i2)
                            
                            u1_val_row = i1*2+j1
                            u2_val_row = i2*3+j2
                            v1_val_row = j1*2+k1
                            v2_val_row = j2*3+k2
                            w1_val_row = k1*2+i1
                            w2_val_row = k2*3+i2
                            
                            for r1 in range(7):
                                u1c = U1[u1_val_row][r1]
                                v1c = V1[v1_val_row][r1]
                                w1c = W1[w1_val_row][r1]
                                for r2 in range(27):
                                    r = r1*27 + r2
                                    U[a_row][r] = u1c * U2[u2_val_row][r2]
                                    V[b_row][r] = v1c * V2[v2_val_row][r2]
                                    W[c_row][r] = w1c * W2[w2_val_row][r2]
    
    return {"U": U, "V": V, "W": W}
