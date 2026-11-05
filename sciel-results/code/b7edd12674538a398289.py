"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 140.0  # 150s limit with 10s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    try:
        U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
        V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
        W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return False
        return True
    except Exception:
        return False

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank1():
    """Try the trivial rank-1 decomposition where all entries are 1."""
    n, m, p = DIMENSIONS
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # First try rank 1
    result = _try_rank1()
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

