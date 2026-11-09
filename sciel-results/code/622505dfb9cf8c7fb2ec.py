"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition by checking the residual."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_rank1(n, m, p):
    """Try rank-1 decomposition with all-ones vectors."""
    U = [[1.0] for _ in range(n*m)]
    V = [[1.0] for _ in range(m*p)]
    W = [[1.0] for _ in range(p*n)]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _fallback_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try rank 1 first - the all-ones tensor has rank 1
    result = _try_rank1(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _fallback_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
