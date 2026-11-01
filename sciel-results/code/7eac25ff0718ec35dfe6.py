"""Rewritten program for matrix multiplication 2,2,10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5-second safety margin

    # Build the target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The matrix multiplication tensor for A(n x m) * B(m x p) -> C(n x p)
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] appears in C[i,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Try rank 1 first: check if T can be written as a single outer product
    # T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k,c
    # For rank 1: T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    # This requires T to be a rank-1 tensor in the (i,j,k) index space
    
    # Check if T is rank 1 by verifying all 2x2 minors are zero
    # Actually, for the all-ones tensor, it IS rank 1:
    # T[i,j,k] = 1 = 1 * 1 * 1
    
    # Let's verify: if U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1 for all indices
    # Then T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i] = 1*1*1 = 1 ✓
    
    # So rank 1 works!
    rank = 1
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    # Verify the decomposition
    # T_decomposed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_decomposed = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    T_decomposed[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    
    if not np.allclose(T, T_decomposed, atol=1e-10):
        # Fallback to schoolbook if rank 1 doesn't work (shouldn't happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
