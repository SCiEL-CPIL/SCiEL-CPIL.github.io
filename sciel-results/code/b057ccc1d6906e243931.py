import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    
    U0 = np.zeros((nm, schoolbook_rank))
    V0 = np.zeros((mp, schoolbook_rank))
    W0 = np.zeros((np_, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
