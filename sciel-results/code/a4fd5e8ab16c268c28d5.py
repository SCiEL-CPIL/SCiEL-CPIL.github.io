import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Use the standard bilinear algorithm: for each (i, k, j), add a rank-1 term
    # where U[i*m + k] = 1, V[k*p + j] = 1, W[j*n + i] = 1
    # This gives rank = n * m * p = 4 * 5 * 5 = 100
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_vec = [0.0] * (n * m)
                v_vec = [0.0] * (m * p)
                w_vec = [0.0] * (p * n)
                
                u_vec[i * m + k] = 1.0
                v_vec[k * p + j] = 1.0
                w_vec[j * n + i] = 1.0
                
                add_rank1(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    if rank == 0:
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[0] = 1.0
        v_vec[0] = 1.0
        w_vec[0] = 1.0
        add_rank1(u_vec, v_vec, w_vec)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
