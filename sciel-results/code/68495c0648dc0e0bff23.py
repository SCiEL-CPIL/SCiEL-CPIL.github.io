"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Use rank 18 schoolbook decomposition as a correct baseline
    # Then attempt to find a rank 17 decomposition if time permits
    # For now, return a verified correct rank 18 solution
    
    rank = 18
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Schoolbook: each term is A[i,j] * B[j,k] -> C[i,k]
    # Index mapping:
    # U: i*m + j for (i,j) in [0,2) x [0,3)
    # V: j*p + k for (j,k) in [0,3) x [0,3)
    # W: k*n + i for (k,i) in [0,3) x [0,2)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, (n, m))
    B_test = np.random.randint(-5, 5, (m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += A_test[i, j] * B_test[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    if not np.allclose(C_expected, C_computed):
        # This should never happen with schoolbook, but if it does, return empty
        return {"U": [], "V": [], "W": []}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
