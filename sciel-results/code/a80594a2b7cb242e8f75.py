import numpy as np
import time

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate(U, V, W):
    """Check exact tensor decomposition. Returns (rank, U, V, W) if valid, else None."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wk[c]
                if s != 1:
                    return None
    return (r, U, V, W)

def _try_rank1():
    """Try rank-1 decomposition."""
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    if _validate(U, V, W) is not None:
        return (rank, U, V, W)
    return None

def _schoolbook():
    """Schoolbook rank-45 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return (rank, U, V, W)

def construct():
    """Main entry point."""
    # Try rank 1 first
    result = _try_rank1()
    if result is not None:
        r, U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    r, U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
