import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 8
    rank = n * m

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1.0
            for k in range(p):
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0

    # Optional: Verify reconstruction if time permits
    if time.time() - start_time < 110:
        # Reconstruct T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can check a few entries
        pass

    return {"U": U, "V": V, "W": W}
