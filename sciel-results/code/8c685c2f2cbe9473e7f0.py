"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]*B[j,k].
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c] should equal sum_j U[i*m+j,c]*V[j*p+k,c].
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,k): sum_j U3[i,j,:] * V3[j,k,:] should equal W3[k,i,:]
    # Wait, the tensor decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let's verify this condition directly as it's the definition of the rank.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to schoolbook
    schoolbook_rank = n * m * p
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
