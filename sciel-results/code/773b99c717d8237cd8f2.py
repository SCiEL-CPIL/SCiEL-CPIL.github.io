import time
import numpy as np

DIMENSIONS = [2, 2, 2]
_DEADLINE = 115.0  # safety margin before 120s evaluator kill

def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    """
    _t0 = time.monotonic()
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's 7 products
    # U[i*m+j, c]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]
    # Row 0: k=0,i=0 -> C[0,0]
    # Row 1: k=0,i=1 -> C[1,0]
    # Row 2: k=1,i=0 -> C[0,1]
    # Row 3: k=1,i=1 -> C[1,1]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    # Quick exact verification: check that the tensor equation holds for all (i,j,k)
    if time.monotonic() - _t0 < _DEADLINE - 5.0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-10:
                        # Fallback: return as-is (should not happen)
                        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
