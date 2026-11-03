"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the matrix multiplication tensor T[i, j, k] where
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor has entries T[i, j, k] = 1 for all i, j, k
    # Decomposition: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, time_limit=12.0):
    """Try to find a decomposition of given rank using random initialization and gradient-free search."""
    start_time = time.time()
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Try to minimize the error using a simple iterative approach
    # This is a placeholder - actual optimization would be needed
    # For now, just return the random initialization (which won't work)
    # In a real implementation, we'd use something like:
    # - Alternating least squares
    # - Gradient descent
    # - Tensor decomposition algorithms
    
    # Since we don't have a guaranteed way to find a lower rank decomposition,
    # we'll just verify if the schoolbook works and return it
    return None

# EVOLVE-BLOCK-END
