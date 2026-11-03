DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    U = [[1,0,1,0,1,-1,1],
         [0,0,0,0,1,0,-1],
         [0,1,0,0,0,1,0],
         [1,1,0,1,0,0,-1]]
    V = [[1,1,0,-1,0,1,0],
         [0,0,1,0,0,1,0],
         [0,0,0,1,0,0,1],
         [1,0,-1,0,1,0,1]]
    W = [[1,0,0,1,-1,0,1],
         [0,0,1,0,1,0,0],
         [0,1,0,1,0,0,0],
         [1,0,0,0,0,1,0]]
    return U, V, W

def _strassen_4x4():
    U2, V2, W2 = _strassen_2x2()
    r = 7
    rank = r * r
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    for c_out in range(r):
        for c_in in range(r):
            col = c_out * r + c_in
            for i in range(4):
                for j in range(4):
                    br, bc = i >> 1, j >> 1
                    ir, ic = i & 1, j & 1
                    U[i*4+j][col] = U2[br*2+bc][c_out] * U2[ir*2+ic][c_in]
            for j in range(4):
                for k in range(4):
                    br, bc = j >> 1, k >> 1
                    ir, ic = j & 1, k & 1
                    V[j*4+k][col] = V2[br*2+bc][c_out] * V2[ir*2+ic][c_in]
            for k in range(4):
                for i in range(4):
                    br, bc = k >> 1, i >> 1
                    ir, ic = k & 1, i & 1
                    W[k*4+i][col] = W2[br*2+bc][c_out] * W2[ir*2+ic][c_in]
    return U, V, W

def _schoolbook_4x4x4x3():
    n, m, p = 4, 4, 3
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U4, V4, W4 = _strassen_4x4()
    U3, V3, W3 = _schoolbook_4x4x4x3()
    r4 = 49
    r3 = 48
    total_rank = r4 + r3
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(16)]
    W = [[0]*total_rank for _ in range(28)]
    for i in range(16):
        for c in range(r4):
            U[i][c] = U4[i][c]
            V[i][c] = V4[i][c]
            W[i][c] = W4[i][c]
    for i in range(16):
        for c in range(r3):
            U[i][r4+c] = U3[i][c]
            V[i][r4+c] = V3[i][c]
    for local_k in range(3):
        for i in range(4):
            for c in range(r3):
                W[(4+local_k)*4 + i][r4+c] = W3[local_k*4 + i][c]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
