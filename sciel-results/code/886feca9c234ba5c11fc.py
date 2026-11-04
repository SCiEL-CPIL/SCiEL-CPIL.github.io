import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 8
    rank = 8
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for t in range(rank):
        s = t // 2  # inner index (0..3)
        a = t % 2   # row index (0..1)
        
        # U: row = a*m + b
        U[a*m + s, t] = 1
        
        # V: row = c*p + d, for all d
        for d in range(p):
            V[s*p + d, t] = 1
        
        # W: row = k*n + i = d'*n + a'
        for dp in range(p):
            W[dp*n + a, t] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
