import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    t0 = time.time()
    deadline = t0 + 140  # 10s safety margin before 150s kill

    # Try quick structured attempts for lower ranks
    for R in [21, 24, 27, 30, 33, 36]:
        if time.time() > deadline - 5:
            break
        sol = _attempt_rank(R, n, m, p)
        if sol is not None:
            return sol

    # Guaranteed rank-63 fallback
    return _trivial_decomposition(n, m, p)

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _attempt_rank(R, n, m, p):
    """Try to find a rank-R decomposition via structured initialization."""
    # For now, return None - structured search not yet implemented
    return None
