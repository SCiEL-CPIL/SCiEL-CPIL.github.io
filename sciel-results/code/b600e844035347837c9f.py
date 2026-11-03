import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication, rank 7."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)*B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A11-A01)*(B00+B01)
    U[3][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for n x m x p matrix multiplication using schoolbook."""
    rank = n * m * p
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

def _embed_U_split_n(U1, U2, n1, n2, m, p, R1, R2):
    n = n1 + n2
    R = R1 + R2
    U = [[0]*R for _ in range(n*m)]
    for i in range(n1*m):
        for r in range(R1):
            U[i][r] = U1[i][r]
    for i in range(n2*m):
        for r in range(R2):
            U[n1*m + i][R1 + r] = U2[i][r]
    return U

def _embed_V_split_n(V1, V2, m, p, R1, R2):
    R = R1 + R2
    V = [[0]*R for _ in range(m*p)]
    for i in range(m*p):
        for r in range(R1):
            V[i][r] = V1[i][r]
        for r in range(R2):
            V[i][R1 + r] = V2[i][r]
    return V

def _embed_W_split_n(W1, W2, p, n1, n2, R1, R2):
    n = n1 + n2
    R = R1 + R2
    W = [[0]*R for _ in range(p*n)]
    for i in range(p*n1):
        for r in range(R1):
            W[i][r] = W1[i][r]
    for i in range(p*n2):
        for r in range(R2):
            W[p*n1 + i][R1 + r] = W2[i][r]
    return W

def _embed_U_split_m(U1, U2, n, m1, m2, p, R1, R2):
    m = m1 + m2
    R = R1 + R2
    U = [[0]*R for _ in range(n*m)]
    for i in range(n):
        for j in range(m1):
            for r in range(R1):
                U[i*m + j][r] = U1[i*m1 + j][r]
        for j in range(m2):
            for r in range(R2):
                U[i*m + m1 + j][R1 + r] = U2[i*m2 + j][r]
    return U

def _embed_V_split_m(V1, V2, m1, m2, p, R1, R2):
    m = m1 + m2
    R = R1 + R2
    V = [[0]*R for _ in range(m*p)]
    for j in range(m1):
        for k in range(p):
            for r in range(R1):
                V[j*p + k][r] = V1[j*p + k][r]
        for j in range(m2):
            for k in range(p):
                for r in range(R2):
                    V[(m1 + j)*p + k][R1 + r] = V2[j*p + k][r]
    return V

def _embed_W_split_m(W1, W2, p, n, R1, R2):
    R = R1 + R2
    W = [[0]*R for _ in range(p*n)]
    for i in range(p*n):
        for r in range(R1):
            W[i][r] = W1[i][r]
        for r in range(R2):
            W[i][R1 + r] = W2[i][r]
    return W

def _embed_U_split_p(U1, U2, n, m, R1, R2):
    R = R1 + R2
    U = [[0]*R for _ in range(n*m)]
    for i in range(n*m):
        for r in range(R1):
            U[i][r] = U1[i][r]
        for r in range(R2):
            U[i][R1 + r] = U2[i][r]
    return U

def _embed_V_split_p(V1, V2, m, p1, p2, R1, R2):
    p = p1 + p2
    R = R1 + R2
    V = [[0]*R for _ in range(m*p)]
    for j in range(m):
        for k in range(p1):
            for r in range(R1):
                V[j*p + k][r] = V1[j*p1 + k][r]
        for k in range(p2):
            for r in range(R2):
                V[j*p + p1 + k][R1 + r] = V2[j*p2 + k][r]
    return V

def _embed_W_split_p(W1, W2, p1, p2, n, R1, R2):
    p = p1 + p2
    R = R1 + R2
    W = [[0]*R for _ in range(p*n)]
    for k in range(p1):
        for i in range(n):
            for r in range(R1):
                W[k*n + i][r] = W1[k*n + i][r]
        for k in range(p2):
            for i in range(n):
                for r in range(R2):
                    W[(p1 + k)*n + i][R1 + r] = W2[k*n + i][r]
    return W

def _decompose(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Heuristic: Split the largest dimension
    # Prefer splitting m (inner) if m > 1, then n, then p
    # Actually, splitting m allows parallel independent calculations for V and W?
    # No, splitting n allows parallel U and W.
    # Splitting m allows parallel U and V.
    # Splitting p allows parallel V and W.
    
    # Let's try splitting m first if m is large
    if m >= 2:
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        R1 = len(V1[0])
        R2 = len(V2[0])
        U = _embed_U_split_m(U1, U2, n, m1, m2, p, R1, R2)
        V = _embed_V_split_m(V1, V2, m1, m2, p, R1, R2)
        W = _embed_W_split_m(W1, W2, p, n, R1, R2)
        return U, V, W
    elif n >= 2:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        R1 = len(V1[0])
        R2 = len(V2[0])
        U = _embed_U_split_n(U1, U2, n1, n2, m, p, R1, R2)
        V = _embed_V_split_n(V1, V2, m, p, R1, R2)
        W = _embed_W_split_n(W1, W2, p, n1, n2, R1, R2)
        return U, V, W
    elif p >= 2:
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        R1 = len(V1[0])
        R2 = len(V2[0])
        U = _embed_U_split_p(U1, U2, n, m, R1, R2)
        V = _embed_V_split_p(V1, V2, m, p1, p2, R1, R2)
        W = _embed_W_split_p(W1, W2, p1, p2, n, R1, R2)
        return U, V, W
    else:
        return _schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 100
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
