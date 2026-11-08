import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n=2, m=3, p=9):
    """Verify that the decomposition is valid."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n=2, m=3, p=9):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _search_rank(R, n=2, m=3, p=9, time_limit=10.0):
    """Try to find a valid rank-R decomposition."""
    start = time.time()
    best = None
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(n*m, R))
        V = np.random.randint(-2, 3, size=(m*p, R))
        W = np.random.randint(-2, 3, size=(p*n, R))
        # Check
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try to find a better decomposition
    for R in range(53, 47, -1):
        if time.time() > deadline - 20:
            break
        result = _search_rank(R, n, m, p, time_limit=5.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
