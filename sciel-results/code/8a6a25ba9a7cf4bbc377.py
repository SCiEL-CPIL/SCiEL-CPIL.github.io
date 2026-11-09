# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p = 27
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify correctness with a random test
    np.random.seed(42)
    A = np.random.randint(1, 10, size=(n, m))
    B = np.random.randint(1, 10, size=(m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p), dtype=int)
    for i in range(n):
        for j in range(p):
            val = 0
            for r in range(rank):
                inner = 0
                for k in range(m):
                    inner += U[i * m + k, r] * V[k * p + j, r]
                val += inner * W[j * n + i, r]
            C_computed[i, j] = val
    
    if not np.array_equal(C_computed, C_expected):
        # Fallback
        return {"U": [[0]*rank for _ in range(n*m)], "V": [[0]*rank for _ in range(m*p)], "W": [[0]*rank for _ in range(p*n)]}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
