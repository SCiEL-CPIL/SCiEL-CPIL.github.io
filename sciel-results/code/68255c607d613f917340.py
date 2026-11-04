import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 4, 5

    # Schoolbook decomposition: rank = n*m*p = 60
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return partial result if time is running out
                    # But since this loop is fast, this shouldn't happen
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # If we broke early, truncate the arrays
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
