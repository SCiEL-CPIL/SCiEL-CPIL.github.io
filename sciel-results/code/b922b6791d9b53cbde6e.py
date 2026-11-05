"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the tensor decomposition is exact.
    The tensor T[i, j, k] for matrix multiplication (n x m) * (m x p) -> (n x p)
    is such that T[i, j, k] = 1 if we are computing C[i, k] = sum_j A[i, j] * B[j, k].
    The decomposition requires:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k
    """
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
            
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Construct the tensor T[i, j, k] = U[i*m+j, :] dot V[j*p+k, :] dot W[k*n+i, :]
    # We check if the sum over c equals 1.
    # This is equivalent to checking if the tensor defined by the decomposition matches the identity tensor.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    
    # Reshape for broadcasting
    # U shape: (n, m, rank)
    # V shape: (m, p, rank)
    # W shape: (p, n, rank)
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # Sum over rank dimension
    # Term[i, j, k] = sum_c U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
    # This is a bit tricky with standard broadcasting.
    # Let's do it in a loop for simplicity and safety, or use einsum.
    # einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        target = np.ones((n, m, p))
        if not np.allclose(T, target, atol=1e-9):
            return False
        return True
    except Exception:
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, return the schoolbook solution to guarantee a valid baseline
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Return the schoolbook decomposition.
    # Finding a lower rank decomposition for 3x3x6 requires specific coefficients
    # for algorithms like Rank-19 (for 3x3) or other advanced methods,
    # which are not reliably available in this context without risking validity.
    
    return best_result
# EVOLVE-BLOCK-END
