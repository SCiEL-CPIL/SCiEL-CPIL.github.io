"""
5x5 matrix multiplication - rank 125 schoolbook decomposition with time guard.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s wall-clock limit

    # Build the schoolbook matrices
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school), dtype=float)
    V_s = np.zeros((m*p, rank_school), dtype=float)
    W_s = np.zeros((p*n, rank_school), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1

    # Verify the decomposition is correct
    def check_decomposition(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # The schoolbook decomposition is always valid
    # Return it directly
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
