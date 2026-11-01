import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Rank 46
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 6 blocks of 2x2
    for b in range(6):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # M1 = (a+d)(e+h)
        # U: a, d
        U[0][col] = 1 # a (0,0)
        U[3][col] = 1 # d (1,1)
        # V: e, h
        V[0 * p + k1][col] = 1 # e (0, k1)
        V[1 * p + k2][col] = 1 # h (1, k2)
        # W: c11, c22
        W[k1 * n + 0][col] = 1 # c (k1, 0)
        W[k2 * n + 1][col] = 1 # c (k2, 1)
        col += 1
        
        # M2 = (c+d)e
        U[2][col] = 1 # c (1,0)
        U[3][col] = 1 # d (1,1)
        V[0 * p + k1][col] = 1 # e
        W[k1 * n + 1][col] = 1 # c (k1, 1)
        col += 1
        
        # M3 = a(f-h)
        U[0][col] = 1 # a
        V[0 * p + k2][col] = 1 # f (0, k2)
        V[1 * p + k2][col] = -1 # h (1, k2)
        W[k2 * n + 0][col] = 1 # c (k2, 0)
        col += 1
        
        # M4 = d(g-e)
        U[3][col] = 1 # d
        V[1 * p + k1][col] = 1 # g (1, k1)
        V[0 * p + k1][col] = -1 # e
        W[k1 * n + 1][col] = 1 # c (k1, 1)
        col += 1
        
        # M5 = (a+b)h
        U[0][col] = 1 # a
        U[1][col] = 1 # b (0,1)
        V[1 * p + k2][col] = 1 # h
        W[k1 * n + 0][col] = -1 # c (k1, 0)
        W[k2 * n + 0][col] = 1 # c (k2, 0)
        col += 1
        
        # M6 = (c-a)(e+f)
        U[2][col] = 1 # c
        U[0][col] = -1 # a
        V[0 * p + k1][col] = 1 # e
        V[0 * p + k2][col] = 1 # f
        W[k2 * n + 1][col] = 1 # c (k2, 1)
        col += 1
        
        # M7 = (b-d)(g+h)
        U[1][col] = 1 # b
        U[3][col] = -1 # d
        V[1 * p + k1][col] = 1 # g
        V[1 * p + k2][col] = 1 # h
        W[k1 * n + 0][col] = 1 # c (k1, 0)
        col += 1
        
    # 1 block of 2x1 (column 12)
    k1 = 12
    # C1 = a e + b g
    # C2 = c e + d g
    
    # P1 = a e
    U[0][col] = 1
    V[0 * p + k1][col] = 1
    W[k1 * n + 0][col] = 1
    col += 1
    
    # P2 = b g
    U[1][col] = 1
    V[1 * p + k1][col] = 1
    W[k1 * n + 0][col] = 1
    col += 1
    
    # P3 = c e
    U[2][col] = 1
    V[0 * p + k1][col] = 1
    W[k1 * n + 1][col] = 1
    col += 1
    
    # P4 = d g
    U[3][col] = 1
    V[1 * p + k1][col] = 1
    W[k1 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
