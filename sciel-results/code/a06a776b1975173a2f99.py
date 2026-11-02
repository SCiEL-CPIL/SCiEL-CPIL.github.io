<<<<<<< SEARCH
U_loc[0, 2] = 1
V_loc[1, 2] = 1; V_loc[3, 2] = -1
W_loc[1, 2] = 1; W_loc[3, 2] = 1
=======
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                U[u_idx, r] = 1
                
                v_idx = j * p + k
                V[v_idx, r] = 1
                
                w_idx = k * n + i
                W[w_idx, r] = 1
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE
