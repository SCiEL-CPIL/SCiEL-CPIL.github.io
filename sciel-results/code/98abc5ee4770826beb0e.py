import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i
    
    # Known rank-17 solution for 2x3x3
    U[ui(0,0), 0] = 1; V[vi(0,0), 0] = 1; W[wi(0,0), 0] = 1
    U[ui(1,1), 0] = 1; V[vi(1,1), 0] = 1; W[wi(1,1), 0] = 1
    
    U[ui(1,0), 1] = 1; U[ui(1,1), 1] = 1
    V[vi(0,0), 1] = 1
    W[wi(0,1), 1] = 1; W[wi(1,1), 1] = -1
    
    U[ui(0,0), 2] = 1
    V[vi(0,1), 2] = 1; V[vi(1,1), 2] = -1
    W[wi(1,0), 2] = 1; W[wi(1,1), 2] = 1
    
    U[ui(1,1), 3] = 1
    V[vi(1,0), 3] = 1; V[vi(0,0), 3] = -1
    W[wi(0,0), 3] = 1; W[wi(0,1), 3] = 1
    
    U[ui(0,0), 4] = 1; U[ui(0,1), 4] = 1
    V[vi(1,1), 4] = 1
    W[wi(0,0), 4] = -1; W[wi(1,0), 4] = 1
    
    U[ui(1,0), 5] = 1; U[ui(0,0), 5] = -1
    V[vi(0,0), 5] = 1; V[vi(0,1), 5] = 1
    W[wi(1,1), 5] = 1
    
    U[ui(0,1), 6] = 1; U[ui(1,1), 6] = -1
    V[vi(1,0), 6] = 1; V[vi(1,1), 6] = 1
    W[wi(0,0), 6] = 1
    
    U[ui(0,0), 7] = 1; V[vi(0,2), 7] = 1; W[wi(2,0), 7] = 1
    U[ui(0,1), 8] = 1; V[vi(1,2), 8] = 1; W[wi(2,0), 8] = 1
    U[ui(1,0), 9] = 1; V[vi(0,2), 9] = 1; W[wi(2,1), 9] = 1
    U[ui(1,1), 10] = 1; V[vi(1,2), 10] = 1; W[wi(2,1), 10] = 1
    U[ui(0,2), 11] = 1; V[vi(2,0), 11] = 1; W[wi(0,0), 11] = 1
    U[ui(0,2), 12] = 1; V[vi(2,1), 12] = 1; W[wi(1,0), 12] = 1
    U[ui(0,2), 13] = 1; V[vi(2,2), 13] = 1; W[wi(2,0), 13] = 1
    U[ui(1,2), 14] = 1; V[vi(2,0), 14] = 1; W[wi(0,1), 14] = 1
    U[ui(1,2), 15] = 1; V[vi(2,1), 15] = 1; W[wi(1,1), 15] = 1
    U[ui(1,2), 16] = 1; V[vi(2,2), 16] = 1; W[wi(2,1), 16] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
