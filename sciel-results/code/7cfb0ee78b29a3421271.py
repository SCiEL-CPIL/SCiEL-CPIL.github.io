import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(np.array(u_vec, dtype=float))
        V_cols.append(np.array(v_vec, dtype=float))
        W_cols.append(np.array(w_vec, dtype=float))
        
    # Trivial rank-1 decomposition for each (i, k, j)
    # This ensures the tensor is exactly reconstructed.
    # Rank will be n * m * p = 100.
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_vec = np.zeros(n * m)
                v_vec = np.zeros(m * p)
                w_vec = np.zeros(p * n)
                
                # U index: i * m + k
                u_vec[i * m + k] = 1.0
                
                # V index: k * p + j
                v_vec[k * p + j] = 1.0
                
                # W index: j * n + i
                w_vec[j * n + i] = 1.0
                
                add_rank1(u_vec, v_vec, w_vec)
                
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
