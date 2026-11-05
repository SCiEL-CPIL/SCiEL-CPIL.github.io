"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 10  # 10s safety margin
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Kronecker product lifting strategy
    # Try to embed 3x3x8 into 4x4x8
    # Use known rank-16 decomposition for M_{4,4,8} via M_{2,2,2} rank-4 Kronecker
    # M_{2,2,2} has rank 4 (Strassen-like)
    # Actually, for 2x2x2, the tensor rank is 7 (Strassen) or 4?
    # The standard tensor rank for matrix multiplication M_{2,2,2} is 7.
    # But if we allow a "relaxed" rank-4 decomposition (which exists in some tensor networks), 
    # then Kronecker product gives rank 16 for M_{4,4,8} (since 4=2x2, 8=2x4).
    # Wait, M_{4,4,8} would be (4*4*8) schoolbook = 128.
    # Let's try to find a lower rank decomposition.
    
    # Alternative: use the known result that M_{3,3,3} has rank 23 (Landsberg).
    # But we need M_{3,3,8}.
    # Let's try to use a direct search or known constructions.
    
    # For now, let's implement a basic tensor decomposition search.
    # This is computationally expensive, so we'll use a time limit.
    
    # Try to find a rank-r decomposition for small r
    # We'll use a randomized approach:
    # 1. Initialize U, V, W randomly
    # 2. Use gradient descent to minimize ||T - sum u_i v_i w_i||^2
    
    # But this is very slow for large tensors.
    # Let's try a different approach: use the known fact that 
    # M_{3,3,8} can be decomposed using the Kronecker product of M_{3,3,3} and M_{1,1,8/3}?
    # That doesn't make sense.
    
    # Let's try to use a fixed-point iteration or other numerical method.
    # But given the time constraint, we might not be able to find a better decomposition.
    
    # For now, return the schoolbook decomposition.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
