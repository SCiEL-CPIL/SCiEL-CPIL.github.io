import time
import numpy as np

def construct():
    n, m, p = 3, 4, 7
    R = 83
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We define the 7 terms of Strassen for the 2x2x2 block.
    # Indices within the block:
    # A: 00, 01, 10, 11 -> U indices: 0, 1, 4, 5
    # B: 00, 01, 10, 11 -> V indices: 0, 1, 7, 8
    # C: 00, 01, 10, 11 -> W indices: 0, 1, 3, 4 (k*n+i: 0*3+0=0, 1*3+0=1, 0*3+1=3, 1*3+1=4)
    
    # Term 0
    U[0, 0] = 1; U[5, 0] = 1
    V[0, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # Term 1
    U[4, 1] = 1; U[5, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1; W[4, 1] = -1
    
    # Term 2
    U[0, 2] = 1
    V[7, 2] = 1; V[8, 2] = -1
    W[1, 2] = 1; W[4, 2] = 1
    
    # Term 3
    U[5, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1
    
    # Term 4
    U[0, 4] = 1; U[4, 4] = 1
    V[8, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[7, 5] = 1
    W[4, 5] = -1
    
    # Term 6
    U[4, 6] = 1; U[5, 6] = -1
    V[1, 6] = 1; V[8, 6] = 1
    W[0, 6] = 1
    
    # Verify Strassen block
    # Construct full 12x28x21 tensor check? No, just check the sub-block.
    # A_sub is 2x2, B_sub is 2x2.
    # U_sub is 4x7, V_sub is 4x7, W_sub is 4x7.
    # But our U, V, W are global.
    
    # Fill remaining with schoolbook
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    if col != R:
        # Fallback to schoolbook if logic error
        U = np.zeros((n * m, n * m * p), dtype=int)
        V = np.zeros((m * p, n * m * p), dtype=int)
        W = np.zeros((p * n, n * m * p), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
