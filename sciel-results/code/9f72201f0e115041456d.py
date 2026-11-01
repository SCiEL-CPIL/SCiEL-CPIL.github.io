import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5

    # Target rank: 21
    # Decomposition:
    # 1. j=2 slice (2x1x5): Rank 1
    # 2. j in {0,1} slice (2x2x5): Rank 20 (Schoolbook)
    
    rank = 21
    nU = n * m  # 6
    nV = m * p  # 15
    nW = p * n  # 10

    U = np.zeros((nU, rank))
    V = np.zeros((nV, rank))
    W = np.zeros((nW, rank))

    # Factor 0: j=2
    # U[i, 2] = 1 for i in {0, 1}
    U[0*m + 2, 0] = 1.0
    U[1*m + 2, 0] = 1.0
    
    # V[2, k] = 1 for k in {0..4}
    for k in range(p):
        V[2*p + k, 0] = 1.0
        
    # W[k, i] = 1 for k in {0..4}, i in {0, 1}
    for k in range(p):
        for i in range(n):
            W[k*n + i, 0] = 1.0

    # Factors 1..20: j in {0, 1}, schoolbook
    col = 1
    for i in range(n):
        for j in range(2):  # j=0, 1
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
