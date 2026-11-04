import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Trivial (schoolbook) decomposition has rank n*m*p
    trivial_rank = n * m * p
    
    U = np.zeros((n * m, trivial_rank))
    V = np.zeros((m * p, trivial_rank))
    W = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                a = i * m + j
                # Index in V: row j*p + k
                b = j * p + k
                # Index in W: row k*n + i (following W[k*n+i, r] convention)
                c = k * n + i
                
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1

    # The task asks for integer/half-integer entries.
    # The trivial decomposition uses 1s and 0s, which are integers.
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
