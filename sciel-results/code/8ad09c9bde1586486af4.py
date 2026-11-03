"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Classic schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W, tol=1e-9):
    """Verify decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1"""
    import numpy as np
    n, m, p = n, m, p
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                prod = u @ v * 0
                # compute sum_c u_c * v_c * w_c
                # can't do elementwise with @
                s = 0.0
                for c in range(rank):
                    s += u[c] * v[c] * w[c]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None


def construct():
    deadline = time.time() + 110  # safety margin within 120s
    n, m, p = DIMENSIONS
    best = _build_schoolbook(n, m, p)
    
    # Schoolbook rank is n*m*p = 75
    # Try to see if we can find a decomposition with rank < 75
    # Using a simple approach: try rank values and check feasibility
    # For now, return schoolbook as the safe baseline
    
    return best

# EVOLVE-BLOCK-END
