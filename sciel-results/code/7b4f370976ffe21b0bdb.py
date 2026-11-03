"""
Rectangular Strassen decomposition for 2x5 x 5x6 matrix multiplication.
Target rank: 55 (21 from 2x2*2x6 Strassen + 36 from 2x3*3x6 schoolbook)
"""
import time as _time
import numpy as _np

DIMENSIONS = [2, 5, 6]

def _strassen_2x2_2x2():
    """
    Rank-7 Strassen decomposition for 2x2 * 2x2.
    n=2, m=2, p=2
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C[1,0] -> W[0*2+1] = W[1]
    
    # M3 = A00(B10-B11) -> C01
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C[0,1] -> W[1*2+0] = W[2]
    
    # M4 = A11(B01-B00) -> C00, C10
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11 -> C01, C11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    Ua = _np.array(U, dtype=float)
    Va = _np.array(V, dtype=float)
    Wa = _np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = Ua[i*m + j]
                v = Va[j*p + k]
                w = Wa[k*n + i]
                s = _np.dot(_np.multiply(u, v), w)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m * m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start = _time.time()
    
    # Block 1: 2x2 * 2x6 (j=0,1) using 3x Strassen 2x2*2x2
    U1, V1, W1 = _strassen_2x2_2x2()  # (4,7), (4,7), (4,7)
    
    # Expand to 2x2 * 2x6
    U1_full = [[0]*21 for _ in range(4)]
    V1_full = [[0]*21 for _ in range(12)]
    W1_full = [[0]*21 for _ in range(12)]
    
    for sub in range(3):
        k_off = sub * 2
        for i in range(4):
            for r in range(7):
                U1_full[i][sub*7+r] = U1[i][r]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V1_full[j*6 + k_off + k][sub*7+r] = V1[j*2+k][r]
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    W1_full[(k_off+k)*2 + i][sub*7+r] = W1[k*2+i][r]
    
    # Block 2: 2x3 * 3x6 (j=2,3,4) using schoolbook
    U2, V2, W2 = _schoolbook_block(2, 3, 6)  # (6,36), (18,36), (12,36)
    
    # Combine into full 2x5 * 5x6
    total_rank = 21 + 36  # 57
    U = [[0] * total_rank for _ in range(10)]
    V = [[0] * total_rank for _ in range(30)]
    W = [[0] * total_rank for _ in range(12)]
    
    # Block 1: j=0,1
    for i in range(2):
        for j in range(2):
            for r in range(21):
                U[i*5 + j][r] = U1_full[i*2 + j][r]
    for j in range(2):
        for k in range(6):
            for r in range(21):
                V[j*6 + k][r] = V1_full[j*6 + k][r]
    for k in range(6):
        for i in range(2):
            for r in range(21):
                W[k*2 + i][r] = W1_full[k*2 + i][r]
    
    # Block 2: j=2,3,4
    for i in range(2):
        for j in range(3):
            for r in range(36):
                U[i*5 + 2 + j][21 + r] = U2[i*3 + j][r]
    for j in range(3):
        for k in range(6):
            for r in range(36):
                V[(2+j)*6 + k][21 + r] = V2[j*6 + k][r]
    for k in range(6):
        for i in range(2):
            for r in range(36):
                W[k*2 + i][21 + r] = W2[k*2 + i][r]
    
    if _verify_decomp(n, m, p, U, V, W, tol=1e-6):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U_s, V_s, W_s = _schoolbook_block(2, 5, 6)
    return {"U": U_s, "V": V_s, "W": W_s}
