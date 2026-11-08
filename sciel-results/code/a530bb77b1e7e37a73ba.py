# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke early, we might have an incomplete matrix, but for 50 it won't break.
    # To be safe, if col < rank, we should probably return a valid partial or just the full one.
    # Since 50 is tiny, it will never break.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
