"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 10  # Safety margin
    
    # Schoolbook decomposition for n x m * m x p
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{ijk} = 1
    # U shape: (n*m, n*m*p)
    # V shape: (m*p, n*m*p)
    # W shape: (p*n, n*m*p)
    
    R = n * m * p
    
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row (i, j) -> i*m + j
                # Index in V: row (j, k) -> j*p + k
                # Index in W: row (k, i) -> k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
                
                if time.time() > deadline:
                    # Return partial if time runs out, though schoolbook is fast
                    return {"U": U[:, :r].tolist(), "V": V[:, :r].tolist(), "W": W[:, :r].tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
