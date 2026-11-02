import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0.0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U: i*2+k -> 0:A00, 1:A01, 2:A10, 3:A11
    V: k*2+j -> 0:B00, 1:B01, 2:B10, 3:B11
    W: j*2+i -> 0:C00, 1:C10, 2:C01, 3:C11
    """
    # Strassen's 7 multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _kronecker_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i1 in range(n1):
        for k1 in range(m1):
            for j1 in range(p1):
                for i2 in range(n2):
                    for k2 in range(m2):
                        for j2 in range(p2):
                            i = i1 * n2 + i2
                            k = k1 * m2 + k2
                            j = j1 * p2 + j2
                            u_idx = i * m + k
                            v_idx = k * p + j
                            w_idx = j * n + i
                            u1_idx = i1 * m1 + k1
                            v1_idx = k1 * p1 + j1
                            w1_idx = j1 * n1 + i1
                            u2_idx = i2 * m2 + k2
                            v2_idx = k2 * p2 + j2
                            w2_idx = j2 * n2 + i2
                            for c1 in range(r1):
                                for c2 in range(r2):
                                    c = c1 * r2 + c2
                                    U[u_idx][c] += U1[u1_idx][c1] * U2[u2_idx][c2]
                                    V[v_idx][c] += V1[v1_idx][c1] * V2[v2_idx][c2]
                                    W[w_idx][c] += W1[w1_idx][c1] * W2[w2_idx][c2]
    return U, V, W

def _schoolbook_nm1(n, m, p=1):
    """Schoolbook for M_{n,m,1}."""
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            U[i * m + k][col] = 1
            V[k * p + 0][col] = 1 # j=0
            W[0 * n + i][col] = 1 # j=0
            col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # 1. Get M_{2,2,2} rank 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # 2. Get M_{4,4,4} rank 49 via Kronecker
    U4, V4, W4 = _kronecker_product(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2)
    
    # 3. Construct M_{4,4,9}
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # Rank = 49 + 49 + 16 = 114
    r1 = len(U4[0]) # 49
    r2 = len(U4[0]) # 49
    r3 = n * m     # 16
    
    total_rank = r1 + r2 + r3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B1 (cols 0-3)
    # U is same
    for i in range(n):
        for k in range(m):
            u_idx = i * m + k
            for c in range(r1):
                U[u_idx][c] = U4[u_idx][c]
    
    # V for B1: V[k*9 + j, c] = V4[k*4 + j, c] for j=0..3
    for k in range(m):
        for j in range(4):
            v_idx = k * p + j
            v4_idx = k * 4 + j
            for c in range(r1):
                V[v_idx][c] = V4[v4_idx][c]
                
    # W for B1: W[j*4 + i, c] = W4[j*4 + i, c] for j=0..3
    for j in range(4):
        for i in range(n):
            w_idx = j * n + i
            w4_idx = j * 4 + i
            for c in range(r1):
                W[w_idx][c] = W4[w4_idx][c]
                
    # Part 2: B2 (cols 4-7)
    offset1 = r1
    for i in range(n):
        for k in range(m):
            u_idx = i * m + k
            for c in range(r2):
                U[u_idx][offset1 + c] = U4[u_idx][c]
                
    # V for B2: V[k*9 + j, offset1+c] = V4[k*4 + (j-4), offset1+c] for j=4..7
    for k in range(m):
        for j in range(4, 8):
            v_idx = k * p + j
            v4_idx = k * 4 + (j - 4)
            for c in range(r2):
                V[v_idx][offset1 + c] = V4[v4_idx][c]
                
    # W for B2: W[j*4 + i, offset1+c] = W4[(j-4)*4 + i, offset1+c] for j=4..7
    for j in range(4, 8):
        for i in range(n):
            w_idx = j * n + i
            w4_idx = (j - 4) * 4 + i
            for c in range(r2):
                W[w_idx][offset1 + c] = W4[w4_idx][c]
                
    # Part 3: B3 (col 8)
    offset2 = r1 + r2
    U3, V3, W3 = _schoolbook_nm1(n, m, 1)
    
    for i in range(n):
        for k in range(m):
            u_idx = i * m + k
            for c in range(r3):
                U[u_idx][offset2 + c] = U3[u_idx][c]
                
    # V for B3: V[k*9 + 8, offset2+c] = V3[k*1 + 0, offset2+c]
    for k in range(m):
        v_idx = k * p + 8
        v3_idx = k * 1 + 0
        for c in range(r3):
            V[v_idx][offset2 + c] = V3[v3_idx][c]
            
    # W for B3: W[8*4 + i, offset2+c] = W3[0*4 + i, offset2+c]
    for i in range(n):
        w_idx = 8 * n + i
        w3_idx = 0 * 4 + i
        for c in range(r3):
            W[w_idx][offset2 + c] = W3[w3_idx][c]
            
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_s = [[0] * rank for _ in range(n * m)]
        V_s = [[0] * rank for _ in range(m * p)]
        W_s = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_s[i * m + k][col] = 1
                    V_s[k * p + j][col] = 1
                    W_s[j * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
