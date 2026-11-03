import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4."""
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U: A[i,j] where i = i1*2+i2, j = j1*2+j2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            # V: B[j,k] where j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            # W: C[i,k] -> W[k*n+i] where k = k1*2+k2, i = i1*2+i2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    
    return U, V, W

def _build_4x4x2_strassen():
    """Build rank-28 decomposition for 4x4x2."""
    n = 4; m = 4; p = 2
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4 block products
    # A00*B00 -> C00
    # A01*B10 -> C01
    # A10*B00 -> C10
    # A11*B10 -> C11
    
    # Product 1: A00 (i<2, j<2) * B00 (j<2, k<2) -> C00 (k<2, i<2)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val
    
    # Product 2: A01 (i<2, j>=2) * B10 (j>=2, k<2) -> C01 (k<2, i<2)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2 + 7 # offset rank
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = 2 + j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 + j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val

    # Product 3: A10 (i>=2, j<2) * B00 (j<2, k<2) -> C10 (k<2, i>=2)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2 + 14
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 + i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = 2 + i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val

    # Product 4: A11 (i>=2, j>=2) * B10 (j>=2, k<2) -> C11 (k<2, i>=2)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2 + 21
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 + i1 * 2 + i2
                            j = 2 + j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[i*m+j][r] += val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 + j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V[j*p+k][r] += val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = 2 + i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W[k*n+i][r] += val

    return U, V, W

def _build_4x4x1_strassen():
    """Build rank-16 decomposition for 4x4x1."""
    n = 4; m = 4; p = 1
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4 block products of 2x2x1
    # A00*b1 -> C00
    # A01*b2 -> C01
    # A10*b1 -> C10
    # A11*b2 -> C11
    
    # 2x2x1 schoolbook
    # C[i] = A[i,0]B[0] + A[i,1]B[1]
    # U[i*2+j, r], V[j, r], W[i, r]
    
    # Product 1: A00 (i<2, j<2) * b1 (j<2) -> C00 (i<2)
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U[i*m+j][r] = 1
            V[j*p][r] = 1
            W[i*n][r] = 1 # k=0, so W[0*4+i] = W[i]
            
    # Product 2: A01 (i<2, j>=2) * b2 (j>=2) -> C01 (i<2)
    for i in range(2):
        for j in range(2):
            r = 4 + i * 2 + j
            U[i*m + (2+j)][r] = 1
            V[(2+j)*p][r] = 1
            W[i*n][r] = 1
            
    # Product 3: A10 (i>=2, j<2) * b1 (j<2) -> C10 (i>=2)
    for i in range(2):
        for j in range(2):
            r = 8 + i * 2 + j
            U[(2+i)*m+j][r] = 1
            V[j*p][r] = 1
            W[(2+i)*n][r] = 1
            
    # Product 4: A11 (i>=2, j>=2) * b2 (j>=2) -> C11 (i>=2)
    for i in range(2):
        for j in range(2):
            r = 12 + i * 2 + j
            U[(2+i)*m + (2+j)][r] = 1
            V[(2+j)*p][r] = 1
            W[(2+i)*n][r] = 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = 49 + 28 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 4x4x4
    U1, V1, W1 = _build_4x4x4_strassen()
    for i in range(n * m):
        for r in range(49):
            U[i][r] += U1[i][r]
    for i in range(m * 4): # m*4 = 16
        for r in range(49):
            # V1 index: j*4+k. Global V index: j*7+k
            j = i // 4
            k = i % 4
            V[j*7+k][r] += V1[i][r]
    for i in range(4 * n): # 4*n = 16
        for r in range(49):
            # W1 index: k*4+i. Global W index: k*4+i
            W[i][r] += W1[i][r]
            
    # 4x4x2
    U2, V2, W2 = _build_4x4x2_strassen()
    for i in range(n * m):
        for r in range(28):
            U[i][49 + r] += U2[i][r]
    for i in range(m * 2): # m*2 = 8
        for r in range(28):
            # V2 index: j*2+k. Global V index: j*7 + (4+k)
            j = i // 2
            k = i % 2
            V[j*7 + (4+k)][49 + r] += V2[i][r]
    for i in range(2 * n): # 2*n = 8
        for r in range(28):
            # W2 index: k*4+i. Global W index: (4+k)*4+i
            k = i // 4
            i_loc = i % 4
            W[(4+k)*4 + i_loc][49 + r] += W2[i][r]
            
    # 4x4x1
    U3, V3, W3 = _build_4x4x1_strassen()
    for i in range(n * m):
        for r in range(16):
            U[i][49 + 28 + r] += U3[i][r]
    for i in range(m * 1): # m*1 = 4
        for r in range(16):
            # V3 index: j. Global V index: j*7 + 6
            j = i
            V[j*7 + 6][49 + 28 + r] += V3[i][r]
    for i in range(1 * n): # 1*n = 4
        for r in range(16):
            # W3 index: i. Global W index: 6*4 + i
            i_loc = i
            W[6*4 + i_loc][49 + 28 + r] += W3[i][r]
            
    return {"U": U, "V": V, "W": W}

