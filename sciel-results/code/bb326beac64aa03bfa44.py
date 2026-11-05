import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # U2, V2, W2 shapes (4, 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook 3x3 factors
    # U3, V3, W3 shapes (9, 27)
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    for i in range(3):
        for j in range(3):
            c = i * 3 + j
            U3[i * 3 + j, c] = 1
            # V3 index depends on j and k. 
            # Wait, in schoolbook, the term is A[i,j] * B[j,k].
            # So for a fixed c (which corresponds to i,j), 
            # U3[i*3+j, c] = 1.
            # V3[j*3+k, c] = 1 for all k? No.
            # The tensor is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
            # For schoolbook, each c corresponds to a unique (i,j,k) tuple?
            # No, rank is n*m*p.
            # So c corresponds to (i,j,k).
            # U[i*m+j, c] = 1 if c maps to (i,j, *).
            # V[j*p+k, c] = 1 if c maps to (*, j, k).
            # W[k*n+i, c] = 1 if c maps to (*, *, i).
            
            # Let's re-read the seed.
            # column = 0
            # for i in range(n):
            #   for j in range(m):
            #     for k in range(p):
            #       U[i*m+j][column] = 1
            #       V[j*p+k][column] = 1
            #       W[k*n+i][column] = 1
            #       column += 1
            
            # So c = i*m*p + j*p + k.
            # U[i*m+j, c] = 1.
            # V[j*p+k, c] = 1.
            # W[k*n+i, c] = 1.
            
            # For 3x3 schoolbook:
            # c = i*9 + j*3 + k.
            # U3[i*3+j, c] = 1.
            # V3[j*3+k, c] = 1.
            # W3[k*3+i, c] = 1.
            
            pass # The loop above is wrong for filling 3x3. Need to iterate k too.

    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i * 9 + j * 3 + k
                U3[i * 3 + j, c] = 1
                V3[j * 3 + k, c] = 1
                W3[k * 3 + i, c] = 1

    # Combine
    rank = 27 * 7
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for r in range(3):
        for s in range(3):
            for t in range(3):
                for u in range(2):
                    for v in range(2):
                        for w in range(2):
                            i = 2 * r + u
                            j = 2 * s + v
                            k = 2 * t + w
                            
                            idx_U = i * 6 + j
                            idx_V = j * 6 + k
                            idx_W = k * 6 + i
                            
                            idx_U3 = r * 3 + s
                            idx_V3 = s * 3 + t
                            idx_W3 = t * 3 + r
                            
                            idx_U2 = u * 2 + v
                            idx_V2 = v * 2 + w
                            idx_W2 = w * 2 + u
                            
                            for c3 in range(27):
                                val3_u = U3[idx_U3, c3]
                                val3_v = V3[idx_V3, c3]
                                val3_w = W3[idx_W3, c3]
                                if val3_u == 0 or val3_v == 0 or val3_w == 0:
                                    continue
                                for c2 in range(7):
                                    val2_u = U2[idx_U2, c2]
                                    val2_v = V2[idx_V2, c2]
                                    val2_w = W2[idx_W2, c2]
                                    if val2_u == 0 or val2_v == 0 or val2_w == 0:
                                        continue
                                    
                                    col = c3 * 7 + c2
                                    U[idx_U, col] += val3_u * val2_u
                                    V[idx_V, col] += val3_v * val2_v
                                    W[idx_W, col] += val3_w * val2_w

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
