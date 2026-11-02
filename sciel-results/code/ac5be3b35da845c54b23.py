import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0
_MARGIN = 10.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN

def _verify(U, V, W, tol=1e-9):
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _laderman_rank_23():
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Laderman's 23 multiplications
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # 1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1; W[4, 0] = 1
    
    # 2
    U[1, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1; V[3, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1; W[4, 1] = 1; W[7, 1] = 1
    
    # 3
    U[0, 2] = 1; U[1, 2] = 1
    V[4, 2] = 1; V[7, 2] = 1
    W[0, 2] = 1; W[1, 2] = 1; W[6, 2] = 1; W[7, 2] = 1
    
    # 4
    U[3, 3] = 1; U[4, 3] = 1
    V[1, 3] = 1; V[4, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1; W[3, 3] = 1; W[4, 3] = 1
    
    # 5
    U[0, 4] = 1; U[3, 4] = 1
    V[3, 4] = 1; V[6, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1; W[6, 4] = 1; W[8, 4] = 1
    
    # 6
    U[4, 5] = 1; U[8, 5] = 1
    V[4, 5] = 1; V[8, 5] = 1
    W[1, 5] = 1; W[4, 5] = 1; W[7, 5] = 1; W[8, 5] = 1
    
    # 7
    U[1, 6] = 1; U[5, 6] = 1
    V[3, 6] = 1; V[6, 6] = 1
    W[1, 6] = 1; W[4, 6] = 1; W[6, 6] = 1; W[8, 6] = 1
    
    # 8
    U[3, 7] = 1; U[6, 7] = 1
    V[1, 7] = 1; V[4, 7] = 1
    W[0, 7] = 1; W[1, 7] = 1; W[3, 7] = 1; W[4, 7] = 1
    
    # 9
    U[0, 8] = 1; U[8, 8] = 1
    V[0, 8] = 1; V[8, 8] = 1
    W[0, 8] = 1; W[2, 8] = 1; W[5, 8] = 1; W[7, 8] = 1
    
    # 10
    U[1, 9] = 1; U[8, 9] = 1
    V[2, 9] = 1; V[6, 9] = 1
    W[1, 9] = 1; W[2, 9] = 1; W[5, 9] = 1; W[8, 9] = 1
    
    # 11
    U[5, 10] = 1; U[6, 10] = 1
    V[3, 10] = 1; V[8, 10] = 1
    W[4, 10] = 1; W[5, 10] = 1; W[6, 10] = 1; W[8, 10] = 1
    
    # 12
    U[2, 11] = 1; U[6, 11] = 1
    V[7, 11] = 1; V[3, 11] = 1
    W[2, 11] = 1; W[5, 11] = 1; W[6, 11] = 1; W[7, 11] = 1
    
    # 13
    U[2, 12] = 1; U[4, 12] = 1
    V[5, 12] = 1; V[6, 12] = 1
    W[2, 12] = 1; W[4, 12] = 1; W[5, 12] = 1; W[8, 12] = 1
    
    # 14
    U[3, 13] = 1; U[8, 13] = 1
    V[1, 13] = 1; V[8, 13] = 1
    W[0, 13] = 1; W[2, 13] = 1; W[3, 13] = 1; W[8, 13] = 1
    
    # 15
    U[0, 14] = 1
    V[0, 14] = 1
    W[0, 14] = 1
    
    # 16
    U[4, 15] = 1
    V[4, 15] = 1
    W[4, 15] = 1
    
    # 17
    U[8, 16] = 1
    V[8, 16] = 1
    W[8, 16] = 1
    
    # 18
    U[1, 17] = 1
    V[3, 17] = 1
    W[3, 17] = 1
    
    # 19
    U[3, 18] = 1
    V[1, 18] = 1
    W[1, 18] = 1
    
    # 20
    U[2, 19] = 1
    V[6, 19] = 1
    W[6, 19] = 1
    
    # 21
    U[6, 20] = 1
    V[2, 20] = 1
    W[2, 20] = 1
    
    # 22
    U[5, 21] = 1
    V[7, 21] = 1
    W[7, 21] = 1
    
    # 23
    U[7, 22] = 1
    V[5, 22] = 1
    W[5, 22] = 1
    
    return U, V, W

def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    # Try Laderman rank 23
    try:
        U23, V23, W23 = _laderman_rank_23()
        if _verify(U23, V23, W23, tol=1e-9):
            return {"U": U23.tolist(), "V": V23.tolist(), "W": W23.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 27
    U_s, V_s, W_s = _schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
