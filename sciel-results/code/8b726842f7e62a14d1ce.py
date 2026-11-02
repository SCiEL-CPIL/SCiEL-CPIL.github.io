import time
import random
import numpy as np

def construct():
    """
    Returns the rank-28 decomposition for (2,2,8) matrix multiplication.
    Uses 4 independent Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16
    
    # Strassen 7-term decomposition for 2x2x2
    # u[i*2+j], v[j*2+k], w[k*2+i]
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for block in range(4):
        for t in range(7):
            col = block * 7 + t
            u, v, w = terms[t]
            
            # U: index a = i*2 + j
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = u[i * 2 + j]
                    
            # V: index b = j*p + k = j*8 + k
            # k_global = 2*block + kl
            for j in range(2):
                for kl in range(2):
                    k_global = 2 * block + kl
                    b_idx = j * p + k_global
                    v_idx = j * 2 + kl
                    V[b_idx][col] = v[v_idx]
                    
            # W: index c = k*n + i = k*2 + i
            # k_global = 2*block + kl
            for i in range(2):
                for kl in range(2):
                    k_global = 2 * block + kl
                    c_idx = k_global * n + i
                    w_idx = kl * 2 + i
                    W[c_idx][col] = w[w_idx]
                    
    return {"U": U, "V": V, "W": W}
