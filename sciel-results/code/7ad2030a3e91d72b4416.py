import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 3, 3
    R = 23
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Strassen Part (7 terms)
    # r=0
    U[idx_u(0, 0), 0] = 1; U[idx_u(1, 1), 0] = 1
    V[idx_v(0, 0), 0] = 1; V[idx_v(1, 1), 0] = 1
    W[idx_w(0, 0), 0] = 1; W[idx_w(1, 1), 0] = 1
    
    # r=1
    U[idx_u(1, 0), 1] = 1; U[idx_u(1, 1), 1] = 1
    V[idx_v(0, 0), 1] = 1
    W[idx_w(0, 1), 1] = 1; W[idx_w(1, 1), 1] = -1
    
    # r=2
    U[idx_u(0, 0), 2] = 1
    V[idx_v(0, 1), 2] = 1; V[idx_v(1, 1), 2] = -1
    W[idx_w(1, 0), 2] = 1; W[idx_w(1, 1), 2] = 1
    
    # r=3
    U[idx_u(1, 1), 3] = 1
    V[idx_v(1, 0), 3] = 1; V[idx_v(0, 0), 3] = -1
    W[idx_w(0, 0), 3] = 1; W[idx_w(0, 1), 3] = 1
    
    # r=4
    U[idx_u(0, 0), 4] = 1; U[idx_u(0, 1), 4] = 1
    V[idx_v(1, 1), 4] = 1
    W[idx_w(0, 0), 4] = -1; W[idx_w(1, 0), 4] = 1
    
    # r=5
    U[idx_u(1, 0), 5] = 1; U[idx_u(0, 0), 5] = -1
    V[idx_v(0, 0), 5] = 1; V[idx_v(0, 1), 5] = 1
    W[idx_w(1, 1), 5] = 1
    
    # r=6
    U[idx_u(0, 1), 6] = 1; U[idx_u(1, 1), 6] = -1
    V[idx_v(1, 0), 6] = 1; V[idx_v(1, 1), 6] = 1
    W[idx_w(0, 0), 6] = 1
    
    # Border Part (16 terms)
    # 3 combined terms
    # Term 7: A[2,0] B[0,0] (C[2,0] + C[2,1])
    U[idx_u(2, 0), 7] = 1
    V[idx_v(0, 0), 7] = 1
    W[idx_w(0, 2), 7] = 1; W[idx_w(1, 2), 7] = 1
    
    # Term 8: A[2,1] B[1,0] (C[2,0] + C[2,1])
    U[idx_u(2, 1), 8] = 1
    V[idx_v(1, 0), 8] = 1
    W[idx_w(0, 2), 8] = 1; W[idx_w(1, 2), 8] = 1
    
    # Term 9: A[2,2] B[2,0] (C[2,0] + C[2,1])
    U[idx_u(2, 2), 9] = 1
    V[idx_v(2, 0), 9] = 1
    W[idx_w(0, 2), 9] = 1; W[idx_w(1, 2), 9] = 1
    
    # 13 single-entry terms
    # Remaining border entries:
    # (2,0,2), (2,1,2), (2,2,2)
    # (0,2,0), (0,2,1), (0,2,2)
    # (1,2,0), (1,2,1), (1,2,2)
    # (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    
    col = 10
    entries = [
        (2, 0, 2), (2, 1, 2), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)
    ]
    
    for i, j, k in entries:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
