import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0
    
    # Rank 60 naive
    rank = 60
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Greedy column removal
    for _ in range(10):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        for r in range(U.shape[1]):
            if time.time() > DEADLINE - 5:
                break
            mask = [i for i in range(U.shape[1]) if i != r]
            if _verify(U[:,mask], V[:,mask], W[:,mask]):
                U, V, W = U[:,mask], V[:,mask], W[:,mask]
                improved = True
                break
        if not improved:
            break
    
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
