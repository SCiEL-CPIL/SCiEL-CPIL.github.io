DIMENSIONS = [4, 4, 4]

def construct():
    # Strassen factors for 2x2 matrix multiplication
    U2 = [[1, 0, 1, 0, 1, -1, 0],
          [0, 0, 0, 0, 1, 0, 1],
          [0, 1, 0, 0, 0, 1, 0],
          [1, 1, 0, 1, 0, 0, -1]]
    
    V2 = [[1, 1, 0, -1, 0, 1, 0],
          [0, 0, 1, 0, 1, 1, 0],
          [0, 0, 0, 1, 0, 0, 1],
          [1, 0, -1, 0, 1, 0, 1]]
    
    W2 = [[1, 0, 0, 1, -1, 0, 1],
          [0, 1, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 1, 0, 0],
          [1, -1, 1, 0, 0, 1, 0]]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U4[(i1*2+i2)*4 + (j1*2+j2), r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            idx_u = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U[idx_u][r] = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
            # V4[(j1*2+j2)*4 + (k1*2+k2), r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            idx_v = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V[idx_v][r] = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
            # W4[(k1*2+k2)*4 + (i1*2+i2), r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            idx_w = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W[idx_w][r] = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
    
    return {"U": U, "V": V, "W": W}
