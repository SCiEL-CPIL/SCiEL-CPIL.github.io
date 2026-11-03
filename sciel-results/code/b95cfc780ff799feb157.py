import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    def verify(U, V, W):
        """Check if decomposition is exact."""
        R = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-10)

    # Fallback: rank 27 schoolbook
    R27 = n * m * p
    U27 = np.zeros((n * m, R27))
    V27 = np.zeros((m * p, R27))
    W27 = np.zeros((p * n, R27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i*m+j, col] = 1
                V27[j*p+k, col] = 1
                W27[k*n+i, col] = 1
                col += 1

    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
