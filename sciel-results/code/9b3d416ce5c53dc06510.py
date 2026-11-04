import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # 1+1+1 partition of middle dimension
    # For each j in {0,1,2} and t in {0,1}:
    #   U[t*m+j, r] = 1
    #   V[j*p+k, r] = 1 for all k
    #   W[k*n+t, r] = 1 for all k
    
    rank = 2 * m  # 2 * 3 = 6
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for j in range(m):
        for t in range(n):
            U[t * m + j, idx] = 1.0
            for k in range(p):
                V[j * p + k, idx] = 1.0
                W[k * n + t, idx] = 1.0
            idx += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
