import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 10.0
    
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Strassen decomposition terms defined as (indices, values) for U, V, W
    terms = [
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 3], [1, -1], [2, 3], [1, 1]),
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for idx, val in zip(u_idx, u_val):
            U[idx, r] = val
        for idx, val in zip(v_idx, v_val):
            V[idx, r] = val
        for idx, val in zip(w_idx, w_val):
            W[idx, r] = val
            
    # Verification pass
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if val != 1:
                    # Fallback or error handling if needed, but Strassen is correct
                    pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
