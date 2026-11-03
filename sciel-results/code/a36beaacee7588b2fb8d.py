import numpy as np
import time

START = time.time()
DEADLINE = 140.0

def _check_time():
    return time.time() >= DEADLINE

def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) as lists of lists.
    U: 4x7, V: 4x7, W: 4x7.
    W index convention: W[k*2 + i] for C[i, k].
    """
    r = 7
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(4)]
    W = [[0]*r for _ in range(4)]
    
    # p1 = a * e
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # p2 = c * f
    U[2][1] = 1; V[1][1] = 1; W[1][1] = 1
    
    # p3 = (a+b) * (f-h)
    U[0][2] = 1; U[1][2] = 1; V[1][2] = 1; V[3][2] = -1; W[2][2] = 1
    
    # p4 = (d-c) * (e+f)
    U[3][3] = 1; U[2][3] = -1; V[0][3] = 1; V[1][3] = 1; W[3][3] = 1
    
    # p5 = d * h
    U[3][4] = 1; V[3][4] = 1; W[3][4] = 1
    
    # p6 = (a-b) * (e+h)
    U[0][5] = 1; U[1][5] = -1; V[0][5] = 1; V[3][5] = 1; W[0][5] = 1
    
    # p7 = (b+c) * (g+e)
    U[1][6] = 1; U[2][6] = 1; V[2][6] = 1; V[0][6] = 1; W[1][6] = 1; W[3][6] = 1
    
    return U, V, W

def _mat4x4x4():
    """
    Rank-49 decomposition for 4x4x4 matrix multiplication.
    Tensor product of two Strassen decompositions.
    """
    U0, V0, W0 = _strassen_2x2()
    r = 49
    U = [[0]*r for _ in range(16)]
    V = [[0]*r for _ in range(16)]
    W = [[0]*r for _ in range(16)]
    
    for p in range(7):
        for q in range(7):
            c = p * 7 + q
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            idx_a = (i1*2 + i2) * 4 + (j1*2 + j2)
                            U[idx_a][c] = U0[i1*2 + j1][p] * U0[i2*2 + j2][q]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            idx_b = (j1*2 + j2) * 4 + (k1*2 + k2)
                            V[idx_b][c] = V0[j1*2 + k1][p] * V0[j2*2 + k2][q]
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            idx_w = (k1*2 + k2) * 4 + (i1*2 + i2)
                            W[idx_w][c] = W0[k1*2 + i1][p] * W0[k2*2 + i2][q]
    return U, V, W

def _mat4x4x2():
    """
    Rank-28 decomposition for 4x4x2 matrix multiplication.
    Tensor product of Strassen (2x2x2) and Schoolbook (2x2x1).
    M_{4,4,2} = M_{2,2,2} \otimes M_{2,2,1}
    """
    U0, V0, W0 = _strassen_2x2()
    # M_{2,2,1} rank 4
    U1 = [[0]*4 for _ in range(4)]
    V1 = [[0]*4 for _ in range(2)]
    W1 = [[0]*4 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            U1[i*2+j][i*2+j] = 1
            V1[j][i*2+j] = 1
            W1[i][i*2+j] = 1
            
    r = 7 * 4 # 28
    U = [[0]*r for _ in range(16)]
    V = [[0]*r for _ in range(8)]
    W = [[0]*r for _ in range(8)]
    
    for p in range(7):
        for q in range(4):
            c = p * 4 + q
            # U: 4x4
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            idx_a = (i1*2 + i2) * 4 + (j1*2 + j2)
                            U[idx_a][c] = U0[i1*2 + j1][p] * U1[i2*2 + j2][q]
            # V: 4x2
            for j1 in range(2):
                for k1 in range(2):
                    # j2 in M_{2,2,1} is the column of B (size 2x1, so only col 0? No, B is 2x1)
                    # Wait, M_{2,2,1} B is 2x1. V1 is 2x4? No, V1 is (m*p, rank) = (2*1, 4) = 2x4.
                    # V1[j, c]. j is 0..1.
                    # In tensor product, V index is (j1, k1) for 4x2.
                    # j1 is from M_{2,2,2} (0..1), k1 is from M_{2,2,1} (0..1).
                    # V1 index is j2 (0..1).
                    idx_b = j1 * 2 + k1
                    V[idx_b][c] = V0[j1*2 + k1][p] * V1[k1][q]
            # W: 2x4 (p=2, n=4) -> W[k*4+i]
            for k1 in range(2):
                for i1 in range(2):
                    # i2 from M_{2,2,1} (0..1)
                    idx_w = k1 * 4 + i1
                    # W1 index is i2 (0..1). W1 is 2x4.
                    W[idx_w][c] = W0[k1*2 + i1][p] * W1[i1][q]
    return U, V, W

def _schoolbook(n, m, p):
    """
    Schoolbook decomposition for n x m times m x p.
    """
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return U, V, W

def construct():
    if _check_time():
        # Fallback schoolbook 120
        n, m, p = 4, 5, 6
        r, U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # 1. A1 (4x4) * B1 (4x4) -> Rank 49
    U1, V1, W1 = _mat4x4x4()
    r1 = 49
    
    # 2. A2 (4x1) * B3 (1x4) -> Rank 16
    U2, V2, W2 = _schoolbook(4, 1, 4)
    r2 = 16
    
    # 3. A1 (4x4) * B2 (4x2) -> Rank 28
    U3, V3, W3 = _mat4x4x2()
    r3 = 28
    
    # 4. A2 (4x1) * B4 (1x2) -> Rank 8
    U4, V4, W4 = _schoolbook(4, 1, 2)
    r4 = 8
    
    total = r1 + r2 + r3 + r4
    n, m, p = 4, 5, 6
    U_g = [[0]*total for _ in range(n*m)]
    V_g = [[0]*total for _ in range(m*p)]
    W_g = [[0]*total for _ in range(p*n)]
    
    base = 0
    # Term 1
    for c in range(r1):
        for a in range(16):
            i = a // 4; j = a % 4
            U_g[i*5 + j][base + c] = U1[a][c]
        for b in range(16):
            j = b // 4; k = b % 4
            V_g[j*6 + k][base + c] = V1[b][c]
        for x in range(16):
            k = x // 4; i = x % 4
            W_g[k*4 + i][base + c] = W1[x][c]
    base += r1
    
    # Term 2
    for c in range(r2):
        for a in range(4):
            i = a; j = 4
            U_g[i*5 + j][base + c] = U2[a][c]
        for b in range(4):
            j = 4; k = b
            V_g[j*6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4; i = x % 4
            W_g[k*4 + i][base + c] = W2[x][c]
    base += r2
    
    # Term 3
    for c in range(r3):
        for a in range(16):
            i = a // 4; j = a % 4
            U_g[i*5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2; k = b % 2
            V_g[j*6 + (k+4)][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4; i = x % 4
            W_g[(k+4)*4 + i][base + c] = W3[x][c]
    base += r3
    
    # Term 4
    for c in range(r4):
        for a in range(4):
            i = a; j = 4
            U_g[i*5 + j][base + c] = U4[a][c]
        for b in range(2):
            j = 4; k = b
            V_g[j*6 + (k+4)][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4; i = x % 4
            W_g[(k+4)*4 + i][base + c] = W4[x][c]
    base += r4
    
    return {"U": U_g, "V": V_g, "W": W_g}
