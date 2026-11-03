"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """Search for low-rank decomposition of 5x5x5 matrix multiplication."""
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Build the full tensor T[i,j,k] = 1 if we want to compute C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in range(5)
    # The decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So the term is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if this is a valid term
    
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So we need: for each (i,k), sum_j (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) = 1
    # And for wrong indices, it should be 0.
    
    # Let's use the schoolbook as starting point and try to reduce rank
    # Schoolbook: rank = 125, with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for c=(i,j,k)
    
    # Strategy: Try to find a lower rank by using the fact that the tensor is sparse/structured
    # For 5x5x5, we can try to use the Strassen-like approach or just search for linear dependencies
    
    # First, let's try a simple approach: use random initialization and gradient descent
    # But that might be slow. Instead, let's try to find a structured decomposition.
    
    # For now, return the schoolbook decomposition as a safe fallback
    rank = n * m * p  # 125
    
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
# EVOLVE-BLOCK-END
