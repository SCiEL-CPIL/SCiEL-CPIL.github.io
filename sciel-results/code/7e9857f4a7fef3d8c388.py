import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in A (flattened row-major)
                idx_u = i * m + j
                # Index in B (flattened row-major)
                idx_v = j * p + k
                # Index in C (flattened, trace convention W[k*n+i, r])
                # Problem states: W[k*n+i, r]
                # This implies C is stored such that index is k*n + i
                idx_w = k * n + i
                
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                r += 1
                
                if time.time() > deadline:
                    # Return partial if needed, but schoolbook is fast
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
