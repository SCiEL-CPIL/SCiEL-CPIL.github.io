"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Schoolbook decomposition: rank = n*m*p = 128
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon = 0.0
                    for r in range(U.shape[1]):
                        recon += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(recon - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Should not happen, but fallback
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
