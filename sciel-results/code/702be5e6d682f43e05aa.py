"""
Matrix multiplication tensor decomposition for 2x4 by 4x7.
Returns a rank-56 schoolbook decomposition, which is guaranteed correct.
The dimension lower bound is 28. A rank-28 solution would be optimal.
This program currently provides the rank-56 baseline.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 150.0
T_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - T_START)


def _build_rank56():
    """
    Schoolbook decomposition: rank = n*m*p = 2*4*7 = 56.
    U[i*m + j, col] = 1
    V[j*p + k, col] = 1
    W[k*n + i, col] = 1
    for each (i, j, k)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(decomp):
    """
    Verify the decomposition.
    Returns True if valid.
    """
    n, m, p = DIMENSIONS
    U = np.array(decomp["U"], dtype=np.int64)
    V = np.array(decomp["V"], dtype=np.int64)
    W = np.array(decomp["W"], dtype=np.int64)
    rank = U.shape[1]
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = int(np.dot(np.dot(U[i * m + j], V[j * p + k]), W[k * n + i]))
                if s != 1:
                    return False
    
    # Check off-diagonal
    for i in range(n):
        for j in range(m):
            for ip in range(n):
                for jp in range(m):
                    if i == ip and j == jp:
                        continue
                    for k in range(p):
                        s = int(np.dot(np.dot(U[i * m + j], V[j * p + k]), W[k * n + ip]))
                        if s != 0:
                            return False
    return True


def construct():
    # Return the guaranteed correct rank-56 solution.
    d = _build_rank56()
    
    # Optional: Verify if time permits (it's fast for rank 56)
    if _time_left() > 10.0:
        if not _verify(d):
            raise ValueError("Verification failed for rank 56 solution")
            
    return d
