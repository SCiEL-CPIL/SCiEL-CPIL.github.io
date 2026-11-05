import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Trivial 3x3x3 (Rank 27)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: 3x3x1 extension (Rank 1)
    # We need T[i,j,3] = 1
    # Use a single rank-1 term: U[i,j]=1, V[j,3]=1, W[3,i]=1
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
    for j in range(m):
        V[j * p + 3, col] = 1
    for i in range(n):
        W[3 * n + i, col] = 1
    col += 1
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
