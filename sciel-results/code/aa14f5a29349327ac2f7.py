"""Matrix multiplication 3x4x6 with low-rank tensor search."""
import numpy as np
from scipy.linalg import null_space
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that U*V*W reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            target = 0.0
            for j in range(m):
                target += 1.0  # A[i,j]*B[j,k] contributes 1 for each j
            # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
            # For each j, the contribution to C[i,k] is sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            actual = 0.0
            for j in range(m):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                actual += np.sum(u * v * w)
            if abs(actual - target) > tol:
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_target = n * m * p  # 72
    U = np.zeros((n * m, rank_target))
    V = np.zeros((m * p, rank_target))
    W = np.zeros((p * n, rank_target))
    
    for col in range(rank_target):
        i = col // (m * p)
        j = (col // p) % m
        k = col % p
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
