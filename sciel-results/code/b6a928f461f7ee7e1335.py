import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start = time.time()
    deadline = start + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[i,j,k] = 1 (matmul tensor)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    def verify(U, V, W, tol=1e-6):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        r = U.shape[1]
        # Reconstruct T[i,j,k]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # 44-rank Strassen-block decomposition
    def strassen_block(A_rows, B_cols, inner_lo, inner_hi):
        # Returns list of (u, v, w) terms for a 2x2x2 block
        # A_rows: 2 row indices of A, B_cols: 2 col indices of B (output), inner: 2 inner indices
        # Actually, let me build the full U, V, W directly.
        pass
    ...
