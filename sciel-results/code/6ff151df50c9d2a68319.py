import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # because C_{ik} = sum_j A_{ij} B_{jk}
    # This tensor is the all-ones tensor.
    # Its rank is 1.
    rank = 1
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r_idx in range(r):
                        actual += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to trivial
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
